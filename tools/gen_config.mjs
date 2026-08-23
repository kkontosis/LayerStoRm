#!/usr/bin/env node
// gen_config.mjs — Generate config_parser.h + config_parser.cpp from schema.json
//
// Usage: node tools/gen_config.mjs [--check]
//   --check: verify generated files match, exit 1 if different (CI mode)

import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { resolve, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const SCHEMA_PATH = resolve(ROOT, 'config/schema.json');
const HEADER_PATH = resolve(ROOT, 'src/config/config_parser.h');
const SOURCE_PATH = resolve(ROOT, 'src/config/config_parser.cpp');

// ── C++ keyword check ──────────────────────────────────────────────────────

const CPP_KEYWORDS = new Set([
  'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor',
  'bool', 'break', 'case', 'catch', 'char', 'char8_t', 'char16_t', 'char32_t',
  'class', 'compl', 'concept', 'const', 'consteval', 'constexpr', 'constinit',
  'const_cast', 'continue', 'co_await', 'co_return', 'co_yield', 'decltype',
  'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
  'explicit', 'export', 'extern', 'false', 'float', 'for', 'friend', 'goto',
  'if', 'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept',
  'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq', 'private',
  'protected', 'public', 'register', 'reinterpret_cast', 'requires', 'return',
  'short', 'signed', 'sizeof', 'static', 'static_assert', 'static_cast',
  'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true',
  'try', 'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using',
  'virtual', 'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq',
]);

function assertNotKeyword(name, context) {
  if (CPP_KEYWORDS.has(name)) {
    throw new Error(
      `C++ keyword "${name}" used as ${context}. ` +
      `Rename it in schema.json (no x-cName support).`
    );
  }
}

// ── Schema loading and $ref resolution ─────────────────────────────────────

const SCHEMA_DIR = dirname(SCHEMA_PATH);
const schema = JSON.parse(readFileSync(SCHEMA_PATH, 'utf8'));
const defs = schema.$defs || {};

// Resolve file-relative $ref for _internal-* sections before main processing.
// Standard JSON Schema mechanism: "$ref": "internal-schemas/foo.schema.json"
// loads the external file and merges its properties into the section.
for (const [key, section] of Object.entries(schema.properties || {})) {
  if (section.$ref && !section.$ref.startsWith('#')) {
    const refFile = resolve(SCHEMA_DIR, section.$ref);
    if (!existsSync(refFile)) {
      throw new Error(`$ref file not found: ${refFile} (referenced by ${key})`);
    }
    const external = JSON.parse(readFileSync(refFile, 'utf8'));
    // Merge: external schema is the base, inline fields override
    const merged = { ...external };
    for (const [k, v] of Object.entries(section)) {
      if (k !== '$ref') merged[k] = v;
    }
    schema.properties[key] = merged;
  }
}

function resolveRef(obj) {
  if (!obj || typeof obj !== 'object') return obj;
  if (obj.$ref) {
    if (!obj.$ref.startsWith('#')) {
      // File-relative $ref for nested properties (already resolved at top level)
      const refFile = resolve(SCHEMA_DIR, obj.$ref);
      if (existsSync(refFile)) {
        const external = JSON.parse(readFileSync(refFile, 'utf8'));
        const merged = { ...external };
        for (const [k, v] of Object.entries(obj)) {
          if (k !== '$ref') merged[k] = v;
        }
        return merged;
      }
    }
    const refPath = obj.$ref.replace('#/$defs/', '');
    const resolved = { ...defs[refPath] };
    for (const [k, v] of Object.entries(obj)) {
      if (k !== '$ref') resolved[k] = v;
    }
    return resolved;
  }
  return obj;
}

// ── Naming helpers ─────────────────────────────────────────────────────────

function snakeToPascal(s) {
  return s.replace(/^_/, '').split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join('');
}

function toSnake(pascalCase) {
  return pascalCase.replace(/([A-Z])/g, '_$1').toLowerCase().replace(/^_/, '');
}

// ── Explicit $defs → C++ name mapping ──────────────────────────────────────

const DEFS_TO_CPP = {
  'gpu_object': 'GpuConfig',
  'rope_scaling_object': 'RopeScalingConfig',
  'cpu_expert_object': 'CpuExpertConfig',
};

// ── Collected types ────────────────────────────────────────────────────────

const enums = [];     // { name, values: string[] }
const structs = [];   // { name, fields: [...] }
const enumSet = new Set();
const structSet = new Set();
const configFileLinks = []; // { sectionKey, filename }

// ── Register an enum if not already registered ─────────────────────────────

// values: array of JSON-string enum values.
// cppNames: optional map { jsonValue: cppIdentifier } (from x-enumNames) used to
//   give a C++-safe member name when the JSON value is a C++ keyword (e.g. "int").
// unsetMember: optional C++-only member name (from x-unsetMember) prepended as
//   the enum's FIRST member. It has no JSON spelling: it never parses, never
//   serializes (to_json skips the field when it holds this value), and becomes
//   the field's default when the schema has none — "absent in JSON" stays
//   distinguishable from every explicit value (used for derived defaults, e.g.
//   memory.arena_attach.on_conflict deriving from persist).
// The enum's stored `values` is an array of { json, cpp } pairs.
function registerEnum(name, values, { external = false, cppNames = null, unsetMember = null } = {}) {
  if (enumSet.has(name)) return;
  const pairs = values.map(v => {
    const cpp = cppNames && cppNames[v] ? cppNames[v] : v;
    if (cppNames && cppNames[v]) {
      assertNotKeyword(cpp, `x-enumNames mapping for "${v}" in ${name}`);
    } else {
      assertNotKeyword(v, `enum value in ${name}`);
    }
    return { json: v, cpp };
  });
  if (unsetMember) assertNotKeyword(unsetMember, `x-unsetMember in ${name}`);
  enums.push({ name, values: pairs, external, unsetMember });
  enumSet.add(name);
}

// ── Sentinel logic ─────────────────────────────────────────────────────────

function getSentinel(prop) {
  const p = resolveRef(prop);
  if (p?.['x-sentinel'] !== undefined) return p['x-sentinel'];
  if (p?.exclusiveMinimum === 0 || (p?.minimum !== undefined && p.minimum >= 1)) return 0;
  if (p?.minimum === 0) return -1;
  return 0;
}

function sentinelLiteral(prop, cppType) {
  const s = getSentinel(prop);
  if (cppType === 'double') return `${s}.0`;
  return `${s}`;
}

// ── Analyze a schema property and return a FieldInfo ────────────────────────

function analyzeField(jsonName, rawProp, parentStructName, requiredList) {
  assertNotKeyword(jsonName, `field name in ${parentStructName}`);
  const prop = resolveRef(rawProp);
  const changeable = prop['x-changeable'] === true || rawProp['x-changeable'] === true;

  // const fields
  if (prop.const !== undefined) {
    return {
      jsonName, cppName: jsonName, cppType: null,
      kind: 'const', constValue: prop.const,
      isRequired: false, defaultLiteral: null, changeable: false,
    };
  }

  // x-internal fields: present in struct but NOT parsed/serialized
  if (prop['x-internal'] === true) {
    const cType = prop['x-cType'] || null;
    const types = getTypes(prop);
    const nonNull = types.filter(t => t !== 'null');
    const baseType = cType || primitiveType(nonNull[0] || prop.type || 'integer');
    const def = prop.default;
    let defLit = null;
    if (def !== undefined && def !== null && !cType) defLit = formatLiteral(def, baseType);
    return {
      jsonName, cppName: jsonName, cppType: baseType,
      kind: 'internal',
      isRequired: false, defaultLiteral: defLit, changeable: false,
    };
  }

  const isRequired = requiredList?.includes(jsonName) ?? false;
  const types = getTypes(prop);
  const hasNull = types.includes('null');
  const nonNull = types.filter(t => t !== 'null');
  const xOverride = prop['x-override'] === true;
  const cType = prop['x-cType'] || null;

  // Handle oneOf with x-cType (variant types: LayerPinSpec, AutoOrInt)
  if (prop.oneOf && cType && (cType === 'LayerPinSpec' || cType === 'AutoOrInt')) {
    return { ...makeVariantField(jsonName, prop, cType, isRequired), changeable };
  }

  // Handle oneOf with null + primitive/$ref (e.g., exit_layer, host_ram_cache_gb)
  if (prop.oneOf) {
    return { ...handleOneOf(jsonName, prop, parentStructName, isRequired), changeable };
  }

  // Enum (string with enum values)
  const enumVals = prop.enum?.filter(v => v !== null && typeof v === 'string');
  if (enumVals?.length > 0 && (nonNull[0] === 'string' || prop.type === 'string')) {
    const enumName = cType || snakeToPascal(jsonName);
    const unsetMember = prop['x-unsetMember'] || null;
    registerEnum(enumName, enumVals, {
      external: prop['x-externalType'] === true,
      cppNames: prop['x-enumNames'] || null,
      unsetMember,
    });
    return {
      jsonName, cppName: jsonName, cppType: enumName,
      kind: 'enum', enumName, unsetMember,
      isSingleValue: enumVals.length === 1,
      isRequired,
      defaultLiteral: defaultForEnum(prop, enumName) ??
          (unsetMember ? `${enumName}::${unsetMember}` : null),
      changeable,
    };
  }

  // Array
  if (nonNull[0] === 'array' || prop.type === 'array') {
    return { ...handleArrayField(jsonName, prop, parentStructName, isRequired, hasNull, xOverride), changeable: false };
  }

  // Nested object
  if (nonNull[0] === 'object' || prop.type === 'object') {
    return { ...handleObjectField(jsonName, prop, parentStructName, isRequired, hasNull, xOverride), changeable: false };
  }

  // $ref that resolves to an object (e.g., rope_scaling → rope_scaling_object)
  if (rawProp.$ref && prop.type === 'object' && prop.properties) {
    const refName = rawProp.$ref.replace('#/$defs/', '');
    const structName = DEFS_TO_CPP[refName] || snakeToPascal(refName);
    ensureStructBuilt(prop, structName);
    const isOpt = !isRequired;
    return {
      jsonName, cppName: jsonName,
      cppType: isOpt ? `std::optional<${structName}>` : structName,
      kind: isOpt ? 'optional_object' : 'object',
      structType: structName,
      isRequired, defaultLiteral: null, changeable: false,
    };
  }

  // Nullable primitive with x-override → plain type with sentinel
  if (hasNull && xOverride) {
    const baseType = primitiveType(nonNull[0] || 'integer');
    return {
      jsonName, cppName: jsonName, cppType: baseType,
      kind: 'x_override', baseType,
      sentinel: getSentinel(prop), sentinelLit: sentinelLiteral(prop, baseType),
      isRequired, defaultLiteral: defaultForXOverride(prop, baseType), changeable,
    };
  }

  // Nullable primitive without x-override → std::optional
  if (hasNull && !xOverride) {
    const baseType = primitiveType(nonNull[0] || 'integer');
    return {
      jsonName, cppName: jsonName, cppType: `std::optional<${baseType}>`,
      kind: 'optional', baseType,
      isRequired, defaultLiteral: null, changeable,
    };
  }

  // Plain primitive
  const baseType = primitiveType(nonNull[0] || prop.type || 'integer');
  const primDef = defaultForPrimitive(prop, baseType);

  // Non-required primitive with no default → std::optional
  if (!isRequired && primDef === null) {
    return {
      jsonName, cppName: jsonName, cppType: `std::optional<${baseType}>`,
      kind: 'optional', baseType,
      isRequired, defaultLiteral: null, changeable,
    };
  }

  return {
    jsonName, cppName: jsonName, cppType: baseType,
    kind: 'primitive',
    isRequired, defaultLiteral: primDef, changeable,
  };
}

function getTypes(prop) {
  if (Array.isArray(prop.type)) return prop.type;
  if (prop.type) return [prop.type];
  if (prop.oneOf) {
    const out = [];
    for (const alt of prop.oneOf) {
      const a = resolveRef(alt);
      if (a.type) out.push(a.type);
      else if (a.const === null || alt.type === 'null') out.push('null');
      else if (a.const !== undefined) out.push(typeof a.const === 'string' ? 'string' : typeof a.const);
    }
    return out;
  }
  return [];
}

function primitiveType(jsonType) {
  switch (jsonType) {
    case 'integer': return 'int';
    case 'number': return 'double';
    case 'boolean': return 'bool';
    case 'string': return 'std::string';
    default: return 'int';
  }
}

// ── Field constructors for specific patterns ───────────────────────────────

function makeVariantField(jsonName, prop, cType, isRequired) {
  const def = prop.default;
  let defLit = null;
  if (typeof def === 'string') defLit = `std::string{"${def}"}`;
  else if (Array.isArray(def)) defLit = `{${def.join(', ')}}`;
  return {
    jsonName, cppName: jsonName, cppType: cType,
    kind: 'variant', isRequired, defaultLiteral: defLit,
  };
}

function handleOneOf(jsonName, prop, parentStructName, isRequired) {
  // Separate null from non-null alternatives
  const alts = prop.oneOf.map(a => resolveRef(a));
  const nonNullAlts = alts.filter(a => a.type !== 'null' && a.const !== null);

  if (nonNullAlts.length === 1) {
    const alt = nonNullAlts[0];
    // null + string
    if (alt.type === 'string') {
      return {
        jsonName, cppName: jsonName, cppType: 'std::optional<std::string>',
        kind: 'optional', baseType: 'std::string',
        isRequired, defaultLiteral: null,
      };
    }
    // null + integer (e.g., exit_layer → oneOf: [null, positive_int])
    if (alt.type === 'integer') {
      return {
        jsonName, cppName: jsonName, cppType: 'std::optional<int>',
        kind: 'optional', baseType: 'int',
        isRequired, defaultLiteral: null,
      };
    }
    // null + object ($ref resolved)
    if (alt.type === 'object' && alt.properties) {
      const structName = snakeToPascal(jsonName) + 'Config';
      ensureStructBuilt(alt, structName);
      return {
        jsonName, cppName: jsonName, cppType: `std::optional<${structName}>`,
        kind: 'optional_object', structType: structName,
        isRequired, defaultLiteral: null,
      };
    }
  }

  // Fallback: treat as first non-null type
  const first = nonNullAlts[0];
  const bt = primitiveType(first?.type || 'integer');
  return {
    jsonName, cppName: jsonName, cppType: `std::optional<${bt}>`,
    kind: 'optional', baseType: bt,
    isRequired, defaultLiteral: null,
  };
}

function handleArrayField(jsonName, prop, parentStructName, isRequired, hasNull, xOverride) {
  const items = resolveRef(prop.items);
  let itemType, itemIsEnum = false, itemEnumName = null;

  // Check raw items for $ref before resolution
  const rawItems = prop.items;
  if (!rawItems) {
    itemType = 'int';
  } else if (rawItems.$ref) {
    // $ref to a $defs entry (e.g., gpu_object)
    const refName = rawItems.$ref.replace('#/$defs/', '');
    const structName = DEFS_TO_CPP[refName] || snakeToPascal(refName);
    const resolved = resolveRef(rawItems);
    if (resolved.type === 'object' && resolved.properties) {
      ensureStructBuilt(resolved, structName);
      itemType = structName;
    } else {
      itemType = primitiveType(resolved.type || 'integer');
    }
  } else {
    const items = resolveRef(rawItems);
    const itemCType = items['x-cType'] || null;
    const itemEnumVals = items.enum?.filter(v => v !== null && typeof v === 'string');
    if (itemEnumVals?.length > 0) {
      itemEnumName = itemCType || snakeToPascal(jsonName.replace(/s$/, ''));
      registerEnum(itemEnumName, itemEnumVals, { cppNames: items['x-enumNames'] || null });
      itemType = itemEnumName;
      itemIsEnum = true;
    } else if (items.type === 'object' && items.properties) {
      const nestedName = snakeToPascal(jsonName) + 'Config';
      ensureStructBuilt(items, nestedName);
      itemType = nestedName;
    } else {
      itemType = primitiveType(items.type || 'integer');
    }
  }

  const cppType = `std::vector<${itemType}>`;
  const def = prop.default;
  let defLit = null;
  if (def !== undefined && def !== null) {
    defLit = formatArrayDefault(def, itemType, itemIsEnum, itemEnumName);
  } else if (hasNull && xOverride) {
    defLit = '{}'; // sentinel = empty
  }

  return {
    jsonName, cppName: jsonName, cppType,
    kind: hasNull && xOverride ? 'x_override_array' : 'array',
    itemType, itemIsEnum, itemEnumName,
    isRequired, defaultLiteral: defLit,
    isNullable: hasNull, isXOverride: xOverride,
  };
}

function formatArrayDefault(def, itemType, isEnum, enumName) {
  if (!Array.isArray(def) || def.length === 0) return '{}';
  const items = def.map(v => {
    if (isEnum && typeof v === 'string') return `${enumName}::${cppEnumMember(enumName, v)}`;
    if (typeof v === 'number') {
      if (itemType === 'double') return v.toString().includes('.') ? v.toString() : v + '.0';
      return v.toString();
    }
    if (typeof v === 'string') return `"${v}"`;
    return v.toString();
  });
  return `{${items.join(', ')}}`;
}

function handleObjectField(jsonName, prop, parentStructName, isRequired, hasNull, xOverride) {
  if (!prop.properties) {
    // Object with no properties — treat as opaque
    return {
      jsonName, cppName: jsonName, cppType: 'nlohmann::json',
      kind: 'primitive', isRequired, defaultLiteral: null,
    };
  }

  let structName;
  if (jsonName === '_advanced') {
    structName = parentStructName.replace(/Config$/, '') + 'AdvancedConfig';
  } else if (jsonName === 'vram_allocation_gb') {
    structName = 'VramAllocationConfig';
  } else if (jsonName.startsWith('_internal-')) {
    structName = snakeToPascal(jsonName.replace(/^_internal-/, '')) + 'InternalConfig';
  } else {
    structName = snakeToPascal(jsonName) + 'Config';
  }

  ensureStructBuilt(prop, structName);

  // Determine if optional: nullable or not required and "semantically optional"
  // Objects that are nullable → std::optional
  // Non-required objects in gpu_object or model section → check if they have no default
  const isOptional = hasNull || (!isRequired && isSemanticOptional(jsonName, parentStructName));

  if (isOptional) {
    return {
      jsonName, cppName: jsonName, cppType: `std::optional<${structName}>`,
      kind: 'optional_object', structType: structName,
      isRequired, defaultLiteral: null,
    };
  }

  return {
    jsonName, cppName: jsonName, cppType: structName,
    kind: 'object', structType: structName,
    isRequired, defaultLiteral: null,
  };
}

// Objects that are std::optional in the original hand-written code
const SEMANTIC_OPTIONAL = new Set([
  'ModelConfig.rope_scaling',
  'ModelConfig.vision',
  'GpuConfig.vram_allocation_gb',
]);

function isSemanticOptional(jsonName, parentStructName) {
  return SEMANTIC_OPTIONAL.has(`${parentStructName}.${jsonName}`);
}

// ── Default value formatters ───────────────────────────────────────────────

function defaultForPrimitive(prop, cppType) {
  const def = prop.default;
  if (def === undefined || def === null) return null;
  return formatLiteral(def, cppType);
}

function defaultForEnum(prop, enumName) {
  const def = prop.default;
  if (def === undefined) return null;
  return `${enumName}::${cppEnumMember(enumName, def)}`;
}

// Map a JSON enum value to its C++ member name for a registered enum,
// honoring any x-enumNames remapping (e.g. "int" → "int_strategy").
function cppEnumMember(enumName, jsonValue) {
  const e = enums.find(x => x.name === enumName);
  const pair = e?.values.find(v => v.json === jsonValue);
  return pair ? pair.cpp : jsonValue;
}

function defaultForXOverride(prop, baseType) {
  const def = prop.default;
  if (def === null || def === undefined) return sentinelLiteral(prop, baseType);
  return formatLiteral(def, baseType);
}

function formatLiteral(val, cppType) {
  if (val === null) return null;
  if (typeof val === 'boolean') return val ? 'true' : 'false';
  if (typeof val === 'number') {
    if (cppType === 'int') return `${val}`;
    const s = val.toString();
    if (s.includes('e') || s.includes('E')) return s;
    if (!s.includes('.')) return s + '.0';
    return s;
  }
  if (typeof val === 'string') {
    if (cppType.startsWith('std::string')) return `"${val}"`;
    return `"${val}"`;
  }
  return null;
}

// ── Ensure a struct is built and registered ────────────────────────────────

function ensureStructBuilt(prop, structName) {
  if (structSet.has(structName)) return;
  structSet.add(structName);

  const fields = [];
  if (prop.properties) {
    for (const [name, rawProp] of Object.entries(prop.properties)) {
      fields.push(analyzeField(name, rawProp, structName, prop.required));
    }
  }
  structs.push({ name: structName, fields });
}

// ── Build all structs from schema ──────────────────────────────────────────

function buildAll() {
  // First build $defs structs
  for (const [defName, defSchema] of Object.entries(defs)) {
    const cppName = DEFS_TO_CPP[defName];
    if (cppName && defSchema.type === 'object' && defSchema.properties) {
      ensureStructBuilt(defSchema, cppName);
    }
  }

  // Build top-level section structs
  const topFields = [];
  const topRequired = schema.required || [];

  for (const [sectionName, rawSection] of Object.entries(schema.properties)) {
    assertNotKeyword(sectionName, 'top-level property');
    const section = resolveRef(rawSection);

    if (section.type === 'object' && section.properties) {
      // Collect x-configFile annotation
      const configFile = section['x-configFile'] || rawSection['x-configFile'];
      if (configFile) {
        configFileLinks.push({ sectionKey: sectionName, filename: configFile });
      }

      // _internal-* sections get a distinct struct name
      let structName;
      if (sectionName.startsWith('_internal-')) {
        const baseName = sectionName.replace(/^_internal-/, '');
        structName = snakeToPascal(baseName) + 'InternalConfig';
      } else {
        structName = snakeToPascal(sectionName) + 'Config';
      }
      ensureStructBuilt(section, structName);
      // Use a C++-safe field name: _internal-prescope → internal_prescope
      const cppName = sectionName.replace(/-/g, '_');
      topFields.push({
        jsonName: sectionName, cppName, cppType: structName,
        kind: 'object', structType: structName,
        isRequired: topRequired.includes(sectionName),
        defaultLiteral: null,
      });
    } else {
      // Non-object top-level field (e.g. preset string)
      topFields.push(analyzeField(sectionName, rawSection, 'Config', topRequired));
    }
  }

  // Config struct
  structSet.add('Config');
  structs.push({ name: 'Config', fields: topFields });
}

// ── Topological sort ───────────────────────────────────────────────────────

function sortStructs() {
  const byName = new Map(structs.map(s => [s.name, s]));
  const visited = new Set();
  const sorted = [];

  function visit(name) {
    if (visited.has(name)) return;
    visited.add(name);
    const s = byName.get(name);
    if (!s) return;
    for (const f of s.fields) {
      if (f.kind === 'const' || !f.cppType) continue;
      const dep = extractStructDep(f.cppType);
      if (dep && byName.has(dep)) visit(dep);
    }
    sorted.push(s);
  }

  for (const s of structs) visit(s.name);
  structs.length = 0;
  structs.push(...sorted);
}

function extractStructDep(cppType) {
  if (!cppType) return null;
  let t = cppType;
  t = t.replace(/^std::optional</, '').replace(/>$/, '');
  t = t.replace(/^std::vector</, '').replace(/>$/, '');
  if (t.endsWith('Config') || t === 'GpuConfig' || t === 'RopeScalingConfig') return t;
  return null;
}

// ── FieldId assignment — assigns a monotonic ID to every leaf field ────────

const fieldEntries = []; // { id, enumName, dottedPath, cppAccessor, field, valueType }

function isLeafKind(kind) {
  return kind === 'primitive' || kind === 'enum' || kind === 'optional' ||
         kind === 'x_override' || kind === 'variant';
}

function cppValueType(field) {
  const t = field.baseType || field.cppType;
  if (t === 'bool') return 0;
  if (t === 'int') return 1;
  if (t === 'double') return 2;
  return -1;
}

function pythonType(field) {
  const t = field.baseType || field.cppType;
  if (t === 'bool') return 'bool';
  if (t === 'int') return 'int';
  if (t === 'double') return 'float';
  return 'str';
}

function assignFieldIds() {
  let nextId = 0;

  function walkStruct(structObj, pathPrefix, accessorPrefix) {
    for (const f of structObj.fields) {
      if (f.kind === 'const' || f.kind === 'internal') continue;

      if (isLeafKind(f.kind)) {
        const dottedPath = pathPrefix ? `${pathPrefix}.${f.jsonName}` : f.jsonName;
        const cppAccessor = accessorPrefix ? `${accessorPrefix}.${f.cppName}` : f.cppName;
        const enumName = 'k' + dottedPath.split('.').map(p =>
          p.replace(/^_/, '').split(/[-_]/).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join('')
        ).join('');

        f.fieldId = nextId;
        fieldEntries.push({
          id: nextId,
          enumName,
          dottedPath,
          cppAccessor: `cfg.${cppAccessor}`,
          field: f,
          valueType: cppValueType(f),
        });
        nextId++;
      } else if (f.kind === 'object' || f.kind === 'optional_object') {
        const childStruct = structs.find(s => s.name === (f.structType || f.cppType.replace('std::optional<', '').replace('>', '')));
        if (childStruct) {
          const childPath = pathPrefix ? `${pathPrefix}.${f.jsonName}` : f.jsonName;
          const childAccessor = accessorPrefix ? `${accessorPrefix}.${f.cppName}` : f.cppName;
          walkStruct(childStruct, childPath, childAccessor);
        }
      }
      // arrays: skip (not individually addressable)
    }
  }

  const configStruct = structs.find(s => s.name === 'Config');
  if (configStruct) {
    walkStruct(configStruct, '', '');
  }
}

const PYTHON_FIELDS_PATH = resolve(ROOT, 'python/orchestrator/_gen_changeable_fields.py');

function generatePythonFields() {
  const L = [];
  L.push('"""AUTO-GENERATED by tools/gen_config.mjs — DO NOT EDIT"""');
  L.push('');
  L.push('CHANGEABLE_FIELDS = [');
  for (const e of fieldEntries) {
    if (!e.field.changeable) continue;
    const pyType = pythonType(e.field);
    const defVal = e.field.defaultLiteral;
    let pyDefault;
    if (defVal === null || defVal === undefined) {
      pyDefault = 'None';
    } else if (defVal === 'true') {
      pyDefault = 'True';
    } else if (defVal === 'false') {
      pyDefault = 'False';
    } else {
      pyDefault = defVal.replace(/\.0$/, '.0');
    }
    L.push(`    (${e.id}, "${e.dottedPath}", ${pyType}, ${pyDefault}),`);
  }
  L.push(']');
  L.push('');
  L.push('FIELD_COUNT = ' + fieldEntries.length);
  L.push('');
  return L.join('\n');
}

// ── Generate header ────────────────────────────────────────────────────────

function generateHeader() {
  const L = [];
  L.push('#pragma once');
  L.push('');
  L.push('// AUTO-GENERATED by tools/gen_config.mjs — DO NOT EDIT');
  L.push('');
  L.push('#include <cstdint>');
  L.push('#include <optional>');
  L.push('#include <string>');
  L.push('#include <variant>');
  L.push('#include <vector>');
  L.push('');
  L.push('#include <nlohmann/json.hpp>');
  L.push('');
  L.push('#include "core/parser_headers.h"');
  L.push('');
  L.push('namespace layerstorm::config {');
  L.push('');
  L.push('// ── Enums ────────────────────────────────────────────────────────────────────');
  L.push('');
  for (const e of enums) {
    if (e.external) continue;  // x-externalType: defined in an external header
    // x-unsetMember: C++-only "absent" sentinel, FIRST so it is the zero value.
    const members = e.unsetMember
      ? [e.unsetMember, ...e.values.map(v => v.cpp)]
      : e.values.map(v => v.cpp);
    L.push(`enum class ${e.name} { ${members.join(', ')} };`);
  }
  L.push('');
  L.push('// ── Special types ────────────────────────────────────────────────────────────');
  L.push('');
  L.push('// "all" or a list of layer indices');
  L.push('using LayerPinSpec = std::variant<std::string, std::vector<int>>;');
  L.push('');
  L.push('// "auto" or a positive integer');
  L.push('using AutoOrInt = std::variant<std::string, int>;');
  L.push('');
  L.push('// ── Structs ──────────────────────────────────────────────────────────────────');
  L.push('');

  for (const s of structs) {
    L.push(`struct ${s.name} {`);
    for (const f of s.fields) {
      if (f.kind === 'const') continue;
      let line = `    ${f.cppType} ${f.cppName}`;
      if (f.defaultLiteral !== null) {
        line += ` = ${f.defaultLiteral}`;
      } else if (f.isRequired && !f.cppType.startsWith('std::optional')) {
        line += '{}';
      }
      line += ';';
      L.push(line);
    }
    L.push('};');
    L.push('');
  }

  L.push('// ── FieldId ──────────────────────────────────────────────────────────────────');
  L.push('');
  L.push('enum class FieldId : uint16_t {');
  for (const e of fieldEntries) {
    L.push(`    ${e.enumName} = ${e.id},`);
  }
  L.push(`    kFieldCount = ${fieldEntries.length}`);
  L.push('};');
  L.push('');

  L.push('// ── Config file-linking ──────────────────────────────────────────────────────');
  L.push('');
  L.push('struct ConfigFileLink {');
  L.push('    const char* section_key;');
  L.push('    const char* filename;');
  L.push('};');
  L.push('');
  L.push('extern const ConfigFileLink kConfigFileLinks[];');
  L.push(`extern const size_t kConfigFileLinkCount;`);
  L.push('');

  L.push('// ── API ──────────────────────────────────────────────────────────────────────');
  L.push('');
  L.push('/// Parse a JSON config file into a strongly-typed Config struct.');
  L.push('/// Throws std::runtime_error on file I/O errors.');
  L.push('/// Throws nlohmann::json::exception on parse/type errors.');
  L.push('Config parse_config(const std::string& path);');
  L.push('');
  L.push('/// Parse a JSON object into a Config struct.');
  L.push('Config parse_config(const nlohmann::json& j);');
  L.push('');
  L.push('/// Serialize a Config back to JSON.');
  L.push('nlohmann::json config_to_json(const Config& cfg);');
  L.push('');
  L.push('/// Returns true if the field can be changed at runtime without engine restart.');
  L.push('bool is_changeable(FieldId id);');
  L.push('');
  L.push('/// Returns the dotted path name of a field (e.g. "prefetch.prescope.top_k").');
  L.push('const char* field_name(FieldId id);');
  L.push('');
  L.push('/// Apply a single field update to the config. Returns true on success.');
  L.push('/// value_type: 0=bool, 1=int32, 2=float. raw_value is reinterpreted.');
  L.push('bool apply_field_update(Config& cfg, FieldId id, uint8_t value_type, uint32_t raw_value);');
  L.push('');
  L.push('}  // namespace layerstorm::config');
  L.push('');
  return L.join('\n');
}

// ── Generate source ────────────────────────────────────────────────────────

function generateSource() {
  const L = [];
  L.push('#include "config_parser.h"');
  L.push('');
  L.push('// AUTO-GENERATED by tools/gen_config.mjs — DO NOT EDIT');
  L.push('');
  L.push('#include <cstring>');
  L.push('#include <fstream>');
  L.push('#include <stdexcept>');
  L.push('#include <string>');
  L.push('');
  L.push('namespace layerstorm::config {');
  L.push('');

  // Enum parsers
  L.push('// ── Enum parsers ─────────────────────────────────────────────────────────────');
  L.push('');
  for (const e of enums) {
    if (e.values.length <= 1) continue;
    const fn = `parse_${toSnake(e.name)}`;
    L.push(`static ${e.name} ${fn}(const std::string& s) {`);
    for (const v of e.values) L.push(`    if (s == "${v.json}") return ${e.name}::${v.cpp};`);
    L.push(`    throw std::runtime_error("Unknown ${toSnake(e.name)}: " + s);`);
    L.push('}');
    L.push('');
    L.push(`static std::string to_string(${e.name} v) {`);
    L.push('    switch (v) {');
    for (const v of e.values) L.push(`        case ${e.name}::${v.cpp}: return "${v.json}";`);
    if (e.unsetMember) L.push(`        case ${e.name}::${e.unsetMember}: break;  // x-unsetMember: no JSON spelling`);
    L.push('    }');
    L.push('    return {};');
    L.push('}');
    L.push('');
  }

  // get_or helper
  L.push('// ── Helper ────────────────────────────────────────────────────────────────────');
  L.push('');
  L.push('template<typename T>');
  L.push('static T get_or(const nlohmann::json& j, const std::string& key, T def) {');
  L.push('    if (j.contains(key) && !j.at(key).is_null()) return j.at(key).get<T>();');
  L.push('    return def;');
  L.push('}');
  L.push('');

  // Special type parsers
  L.push('// ── Special type parsers ──────────────────────────────────────────────────────');
  L.push('');
  emitVariantParsers(L);

  // Section parsers
  L.push('// ── Section parsers ──────────────────────────────────────────────────────────');
  L.push('');
  for (const s of structs) {
    if (s.name === 'Config') continue;
    emitStructParser(L, s);
    emitStructToJson(L, s);
  }

  // Top-level Config
  const cfg = structs.find(s => s.name === 'Config');
  L.push('// ── Public API ───────────────────────────────────────────────────────────────');
  L.push('');
  L.push('Config parse_config(const nlohmann::json& j) {');
  L.push('    Config cfg;');
  const topReq = schema.required || [];
  for (const f of cfg.fields) {
    if (f.kind === 'const') continue;
    if (f.kind === 'object') {
      // Section struct — use named parser
      const parser = parserNameFor(f.structType || f.cppType);
      if (topReq.includes(f.jsonName)) {
        L.push(`    cfg.${f.cppName} = ${parser}(j.at("${f.jsonName}"));`);
      } else {
        L.push(`    if (j.contains("${f.jsonName}")) cfg.${f.cppName} = ${parser}(j.at("${f.jsonName}"));`);
      }
    } else {
      // Non-object top-level field (e.g. preset string)
      emitFieldParse(L, f, 'cfg', 'j');
    }
  }
  L.push('    return cfg;');
  L.push('}');
  L.push('');

  L.push('Config parse_config(const std::string& path) {');
  L.push('    std::ifstream f(path);');
  L.push('    if (!f.is_open())');
  L.push('        throw std::runtime_error("Cannot open config file: " + path);');
  L.push('    nlohmann::json j = nlohmann::json::parse(f);');
  L.push('    return parse_config(j);');
  L.push('}');
  L.push('');

  L.push('nlohmann::json config_to_json(const Config& cfg) {');
  L.push('    nlohmann::json j;');
  for (const f of cfg.fields) {
    if (f.kind === 'const') continue;
    if (f.kind === 'object') {
      const toJson = toJsonNameFor(f.structType || f.cppType);
      L.push(`    j["${f.jsonName}"] = ${toJson}(cfg.${f.cppName});`);
    } else {
      emitFieldToJson(L, f, 'cfg');
    }
  }
  L.push('    return j;');
  L.push('}');
  L.push('');

  // is_changeable()
  L.push('// ── FieldId helpers ──────────────────────────────────────────────────────────');
  L.push('');
  L.push('bool is_changeable(FieldId id) {');
  L.push('    switch (id) {');
  for (const e of fieldEntries) {
    if (e.field.changeable) {
      L.push(`        case FieldId::${e.enumName}: return true;`);
    }
  }
  L.push('        default: return false;');
  L.push('    }');
  L.push('}');
  L.push('');

  // field_name()
  L.push('const char* field_name(FieldId id) {');
  L.push('    switch (id) {');
  for (const e of fieldEntries) {
    L.push(`        case FieldId::${e.enumName}: return "${e.dottedPath}";`);
  }
  L.push('        default: return "<unknown>";');
  L.push('    }');
  L.push('}');
  L.push('');

  // apply_field_update()
  L.push('bool apply_field_update(Config& cfg, FieldId id, uint8_t value_type, uint32_t raw_value) {');
  L.push('    if (!is_changeable(id)) return false;');
  L.push('    switch (id) {');
  for (const e of fieldEntries) {
    if (!e.field.changeable) continue;
    if (e.valueType === 0) {
      // bool
      L.push(`        case FieldId::${e.enumName}:`);
      L.push(`            if (value_type != 0) return false;`);
      L.push(`            ${e.cppAccessor} = raw_value != 0;`);
      L.push(`            return true;`);
    } else if (e.valueType === 1) {
      // int
      L.push(`        case FieldId::${e.enumName}:`);
      L.push(`            if (value_type != 1) return false;`);
      L.push(`            ${e.cppAccessor} = static_cast<int>(raw_value);`);
      L.push(`            return true;`);
    } else if (e.valueType === 2) {
      // float/double — reinterpret uint32_t as float, then assign to double
      L.push(`        case FieldId::${e.enumName}: {`);
      L.push(`            if (value_type != 2) return false;`);
      L.push(`            float fv; std::memcpy(&fv, &raw_value, sizeof(float));`);
      L.push(`            ${e.cppAccessor} = static_cast<double>(fv);`);
      L.push(`            return true;`);
      L.push(`        }`);
    }
  }
  L.push('        default: return false;');
  L.push('    }');
  L.push('}');
  L.push('');

  // config file-link table
  L.push('// ── Config file-link table (auto-generated from x-configFile) ────────────────');
  L.push('');
  L.push('const ConfigFileLink kConfigFileLinks[] = {');
  if (configFileLinks.length === 0) {
    L.push('    {"", ""},  // sentinel — no file-linked sections');
  } else {
    for (const { sectionKey, filename } of configFileLinks) {
      L.push(`    {"${sectionKey}", "${filename}"},`);
    }
  }
  L.push('};');
  L.push(`const size_t kConfigFileLinkCount = ${configFileLinks.length};`);
  L.push('');

  L.push('}  // namespace layerstorm::config');
  L.push('');
  return L.join('\n');
}

function parserNameFor(typeName) {
  const base = typeName.replace(/Config$/, '');
  return `parse_${toSnake(base)}`;
}

function toJsonNameFor(typeName) {
  const base = typeName.replace(/Config$/, '');
  return `${toSnake(base)}_to_json`;
}

function emitVariantParsers(L) {
  L.push('static LayerPinSpec parse_layer_pin_spec(const nlohmann::json& j) {');
  L.push('    if (j.is_string()) {');
  L.push('        std::string s = j.get<std::string>();');
  L.push('        if (s != "all")');
  L.push('            throw std::runtime_error("layer_pin_spec string must be \\"all\\", got: " + s);');
  L.push('        return s;');
  L.push('    }');
  L.push('    if (j.is_array()) {');
  L.push('        std::vector<int> v;');
  L.push('        for (const auto& el : j) v.push_back(el.get<int>());');
  L.push('        return v;');
  L.push('    }');
  L.push('    throw std::runtime_error("layer_pin_spec must be \\"all\\" or array of integers");');
  L.push('}');
  L.push('');
  L.push('static nlohmann::json layer_pin_spec_to_json(const LayerPinSpec& spec) {');
  L.push('    if (std::holds_alternative<std::string>(spec)) return std::get<std::string>(spec);');
  L.push('    return std::get<std::vector<int>>(spec);');
  L.push('}');
  L.push('');
  L.push('static AutoOrInt parse_auto_or_int(const nlohmann::json& j) {');
  L.push('    if (j.is_string()) {');
  L.push('        std::string s = j.get<std::string>();');
  L.push('        if (s != "auto")');
  L.push('            throw std::runtime_error("auto_or_int string must be \\"auto\\", got: " + s);');
  L.push('        return s;');
  L.push('    }');
  L.push('    if (j.is_number_integer()) return j.get<int>();');
  L.push('    throw std::runtime_error("auto_or_int must be \\"auto\\" or a positive integer");');
  L.push('}');
  L.push('');
  L.push('static nlohmann::json auto_or_int_to_json(const AutoOrInt& v) {');
  L.push('    if (std::holds_alternative<std::string>(v)) return std::get<std::string>(v);');
  L.push('    return std::get<int>(v);');
  L.push('}');
  L.push('');
}

function emitStructParser(L, s) {
  const fn = parserNameFor(s.name);
  L.push(`static ${s.name} ${fn}(const nlohmann::json& j) {`);
  L.push(`    ${s.name} r;`);

  for (const f of s.fields) {
    if (f.kind === 'const' || f.kind === 'internal') continue;
    emitFieldParse(L, f, 'r', 'j');
  }

  L.push('    return r;');
  L.push('}');
  L.push('');
}

function emitFieldParse(L, f, v, j) {
  const acc = `${v}.${f.cppName}`;
  const key = f.jsonName;

  switch (f.kind) {
    case 'variant':
      if (f.cppType === 'LayerPinSpec')
        L.push(`    if (${j}.contains("${key}")) ${acc} = parse_layer_pin_spec(${j}.at("${key}"));`);
      else if (f.cppType === 'AutoOrInt')
        L.push(`    if (${j}.contains("${key}")) ${acc} = parse_auto_or_int(${j}.at("${key}"));`);
      break;

    case 'enum':
      if (f.isSingleValue) break; // const enum — skip
      if (f.isRequired) {
        L.push(`    ${acc} = parse_${toSnake(f.enumName)}(${j}.at("${key}").get<std::string>());`);
      } else {
        L.push(`    if (${j}.contains("${key}")) ${acc} = parse_${toSnake(f.enumName)}(${j}.at("${key}").get<std::string>());`);
      }
      break;

    case 'array':
    case 'x_override_array':
      if (f.itemIsEnum) {
        L.push(`    if (${j}.contains("${key}") && !${j}.at("${key}").is_null()) {`);
        L.push(`        ${acc}.clear();`);
        L.push(`        for (const auto& el : ${j}.at("${key}")) ${acc}.push_back(parse_${toSnake(f.itemEnumName)}(el.get<std::string>()));`);
        L.push('    }');
      } else if (f.itemType.endsWith('Config')) {
        // Array of structs
        const itemParser = parserNameFor(f.itemType);
        if (f.isRequired) {
          L.push(`    for (const auto& el : ${j}.at("${key}")) ${acc}.push_back(${itemParser}(el));`);
        } else {
          L.push(`    if (${j}.contains("${key}")) {`);
          L.push(`        for (const auto& el : ${j}.at("${key}")) ${acc}.push_back(${itemParser}(el));`);
          L.push('    }');
        }
      } else {
        L.push(`    if (${j}.contains("${key}") && !${j}.at("${key}").is_null()) {`);
        L.push(`        ${acc}.clear();`);
        L.push(`        for (const auto& el : ${j}.at("${key}")) ${acc}.push_back(el.get<${f.itemType}>());`);
        L.push('    }');
      }
      break;

    case 'object':
      L.push(`    if (${j}.contains("${key}")) ${acc} = ${parserNameFor(f.structType)}(${j}.at("${key}"));`);
      break;

    case 'optional_object':
      L.push(`    if (${j}.contains("${key}") && !${j}.at("${key}").is_null()) ${acc} = ${parserNameFor(f.structType)}(${j}.at("${key}"));`);
      break;

    case 'x_override': {
      const sent = f.sentinelLit;
      if (f.isRequired) {
        L.push(`    ${acc} = ${j}.at("${key}").is_null() ? ${f.baseType}(${sent}) : ${j}.at("${key}").get<${f.baseType}>();`);
      } else {
        L.push(`    if (${j}.contains("${key}")) ${acc} = ${j}.at("${key}").is_null() ? ${f.baseType}(${sent}) : ${j}.at("${key}").get<${f.baseType}>();`);
      }
      break;
    }

    case 'optional':
      L.push(`    if (${j}.contains("${key}") && !${j}.at("${key}").is_null()) ${acc} = ${j}.at("${key}").get<${f.baseType}>();`);
      break;

    case 'primitive':
      if (f.isRequired) {
        L.push(`    ${acc} = ${j}.at("${key}").get<${f.cppType}>();`);
      } else if (f.defaultLiteral !== null) {
        if (f.cppType === 'std::string') {
          const strDef = f.defaultLiteral.startsWith('"') ? `std::string{${f.defaultLiteral}}` : f.defaultLiteral;
          L.push(`    ${acc} = get_or(${j}, "${key}", ${strDef});`);
        } else {
          L.push(`    ${acc} = get_or(${j}, "${key}", ${f.defaultLiteral});`);
        }
      } else {
        L.push(`    if (${j}.contains("${key}")) ${acc} = ${j}.at("${key}").get<${f.cppType}>();`);
      }
      break;
  }
}

function emitStructToJson(L, s) {
  const fn = toJsonNameFor(s.name);
  L.push(`static nlohmann::json ${fn}(const ${s.name}& r) {`);
  L.push('    nlohmann::json j;');

  for (const f of s.fields) {
    if (f.kind === 'internal') continue;
    if (f.kind === 'const') {
      if (typeof f.constValue === 'string') L.push(`    j["${f.jsonName}"] = "${f.constValue}";`);
      else if (typeof f.constValue === 'boolean') L.push(`    j["${f.jsonName}"] = ${f.constValue};`);
      else L.push(`    j["${f.jsonName}"] = ${f.constValue};`);
      continue;
    }
    emitFieldToJson(L, f, 'r');
  }

  L.push('    return j;');
  L.push('}');
  L.push('');
}

function emitFieldToJson(L, f, v) {
  const acc = `${v}.${f.cppName}`;
  const key = f.jsonName;

  switch (f.kind) {
    case 'variant':
      if (f.cppType === 'LayerPinSpec') L.push(`    j["${key}"] = layer_pin_spec_to_json(${acc});`);
      else if (f.cppType === 'AutoOrInt') L.push(`    j["${key}"] = auto_or_int_to_json(${acc});`);
      break;

    case 'enum':
      if (f.isSingleValue) break;
      if (f.unsetMember) {
        // x-unsetMember: "absent" round-trips as absent
        L.push(`    if (${acc} != ${f.enumName}::${f.unsetMember}) j["${key}"] = to_string(${acc});`);
      } else {
        L.push(`    j["${key}"] = to_string(${acc});`);
      }
      break;

    case 'array':
      if (f.itemIsEnum) {
        L.push(`    { nlohmann::json arr = nlohmann::json::array(); for (const auto& el : ${acc}) arr.push_back(to_string(el)); j["${key}"] = arr; }`);
      } else if (f.itemType.endsWith('Config')) {
        const fn = toJsonNameFor(f.itemType);
        L.push(`    { nlohmann::json arr = nlohmann::json::array(); for (const auto& el : ${acc}) arr.push_back(${fn}(el)); j["${key}"] = arr; }`);
      } else {
        L.push(`    j["${key}"] = ${acc};`);
      }
      break;

    case 'x_override_array':
      L.push(`    j["${key}"] = ${acc}.empty() ? nlohmann::json(nullptr) : nlohmann::json(${acc});`);
      break;

    case 'object':
      L.push(`    j["${key}"] = ${toJsonNameFor(f.structType)}(${acc});`);
      break;

    case 'optional_object': {
      const fn = toJsonNameFor(f.structType);
      L.push(`    if (${acc}) j["${key}"] = ${fn}(*${acc}); else j["${key}"] = nullptr;`);
      break;
    }

    case 'x_override':
      if (f.baseType === 'double') {
        L.push(`    j["${key}"] = ${acc} <= 0.0 ? nlohmann::json(nullptr) : nlohmann::json(${acc});`);
      } else {
        const sentinel = f.sentinel;
        if (sentinel < 0) {
          L.push(`    j["${key}"] = ${acc} < 0 ? nlohmann::json(nullptr) : nlohmann::json(${acc});`);
        } else {
          L.push(`    j["${key}"] = ${acc} == 0 ? nlohmann::json(nullptr) : nlohmann::json(${acc});`);
        }
      }
      break;

    case 'optional':
      L.push(`    j["${key}"] = ${acc} ? nlohmann::json(*${acc}) : nlohmann::json(nullptr);`);
      break;

    case 'primitive':
      L.push(`    j["${key}"] = ${acc};`);
      break;
  }
}

// ── Example config generation ───────────────────────────────────────────────

const EXAMPLE_PATH = resolve(ROOT, 'config/config-example.json');
const PRESETS_DIR = resolve(ROOT, 'config/presets');

// Hardcoded example values for required fields with no schema default.
// These are DeepSeek V3.2 NVFP4 values — the canonical target model.
const EXAMPLE_OVERRIDES = {
  'model.architecture':           'deepseek_v3',
  'model.weights_path':           '/data/models/deepseek-v3.2-nvfp4/',
  'model.weights_format':         'safetensors',
  'model.num_hidden_layers':      61,
  'model.hidden_size':            7168,
  'model.num_attention_heads':    128,
  'model.num_key_value_heads':    128,
  'model.intermediate_size':      18432,
  'model.n_routed_experts':       256,
  'model.num_experts_per_tok':    8,
  'model.vocab_size':             129280,
  'model.max_position_embeddings': 163840,
  // hardware.gpus: full array (id + type only; x-override sub-fields skipped)
  'hardware.gpus': [
    { 'id': 0, 'type': 'rtx5090' },
    { 'id': 1, 'type': 'rtx5090' },
    { 'id': 2, 'type': 'rtx5080' },
    { 'id': 3, 'type': 'rtx5080' },
  ],
};

function buildExampleObject(properties, path) {
  const obj = {};
  for (const [key, rawProp] of Object.entries(properties)) {
    const childPath = path ? `${path}.${key}` : key;
    const prop = resolveRef(rawProp);

    // Skip x-internal and x-override fields
    if (prop['x-internal'] || prop['x-override']) continue;

    // If the whole property is in EXAMPLE_OVERRIDES, use it directly
    if (Object.prototype.hasOwnProperty.call(EXAMPLE_OVERRIDES, childPath)) {
      obj[key] = EXAMPLE_OVERRIDES[childPath];
      continue;
    }

    // Recurse into nested objects
    if (prop.type === 'object' && prop.properties) {
      const child = buildExampleObject(prop.properties, childPath);
      if (child !== null) obj[key] = child;
      continue;
    }

    // Use schema default if present
    if (Object.prototype.hasOwnProperty.call(prop, 'default')) {
      if (prop.default !== null) obj[key] = prop.default;
      continue;
    }

    // oneOf with null → optional, omit
    if (prop.oneOf) continue;

    // No default, not in overrides → omit (optional field)
  }
  return Object.keys(obj).length ? obj : null;
}

function generateExample() {
  const topProps = schema.properties || {};
  const result = buildExampleObject(topProps, '');
  return JSON.stringify(result, null, 2) + '\n';
}

// ── Preset validation ──────────────────────────────────────────────────────

const FORBIDDEN_PRESET_SECTIONS = ['model', 'quantization', 'hardware', 'preset'];

function validatePresetValue(value, schemaProp, path) {
  const prop = resolveRef(schemaProp);
  const errors = [];

  if (value === null) return errors; // null is allowed (merge_patch semantics)

  if (typeof value === 'object' && !Array.isArray(value)) {
    // Recurse into nested objects
    if (prop.type !== 'object' || !prop.properties) {
      errors.push(`${path}: schema says this is not an object with properties`);
      return errors;
    }
    for (const [k, v] of Object.entries(value)) {
      if (!(k in prop.properties)) {
        errors.push(`${path}.${k}: not a known schema property`);
      } else {
        errors.push(...validatePresetValue(v, prop.properties[k], `${path}.${k}`));
      }
    }
    return errors;
  }

  // Leaf value — check type compatibility
  const resolvedProp = resolveRef(schemaProp);
  const schemaTypes = getTypes(resolvedProp);
  const jsType = typeof value;

  if (Array.isArray(value)) {
    if (!schemaTypes.includes('array') && resolvedProp.type !== 'array') {
      errors.push(`${path}: expected non-array, got array`);
    }
    return errors;
  }

  // Check enum constraints
  const enumVals = resolvedProp.enum;
  if (enumVals && !enumVals.includes(value)) {
    errors.push(`${path}: value "${value}" not in enum [${enumVals.join(', ')}]`);
  }

  // Type check (loose: JSON number covers both integer and number)
  if (jsType === 'string' && !schemaTypes.includes('string') && resolvedProp.type !== 'string') {
    // Check oneOf for string alternative (e.g. LayerPinSpec "all", AutoOrInt "auto")
    const hasStringAlt = resolvedProp.oneOf?.some(alt => alt.const && typeof alt.const === 'string');
    if (!hasStringAlt) errors.push(`${path}: expected non-string, got string`);
  }
  if (jsType === 'number' && !schemaTypes.includes('number') && !schemaTypes.includes('integer')
      && resolvedProp.type !== 'number' && resolvedProp.type !== 'integer') {
    errors.push(`${path}: expected non-number, got number`);
  }
  if (jsType === 'boolean' && !schemaTypes.includes('boolean') && resolvedProp.type !== 'boolean') {
    errors.push(`${path}: expected non-boolean, got boolean`);
  }

  return errors;
}

function validatePresets() {
  if (!existsSync(PRESETS_DIR)) {
    console.log('No presets directory found, skipping validation');
    return true;
  }

  const files = readdirSync(PRESETS_DIR).filter(f => f.endsWith('.json'));
  if (files.length === 0) {
    console.log('No preset files found, skipping validation');
    return true;
  }

  let allOk = true;
  const topProps = schema.properties || {};

  for (const file of files) {
    const path = resolve(PRESETS_DIR, file);
    const name = basename(file, '.json');
    let preset;
    try {
      preset = JSON.parse(readFileSync(path, 'utf8'));
    } catch (e) {
      console.error(`${file}: malformed JSON — ${e.message}`);
      allOk = false;
      continue;
    }

    if (typeof preset !== 'object' || Array.isArray(preset) || preset === null) {
      console.error(`${file}: preset must be a JSON object`);
      allOk = false;
      continue;
    }

    const errors = [];

    // Check forbidden sections
    for (const sec of FORBIDDEN_PRESET_SECTIONS) {
      if (sec in preset) {
        errors.push(`contains forbidden key "${sec}"`);
      }
    }

    // Validate every key path against schema
    for (const [key, value] of Object.entries(preset)) {
      if (FORBIDDEN_PRESET_SECTIONS.includes(key)) continue; // already reported
      if (!(key in topProps)) {
        errors.push(`"${key}": not a known top-level config section`);
      } else {
        errors.push(...validatePresetValue(value, topProps[key], key));
      }
    }

    if (errors.length > 0) {
      console.error(`${file}:`);
      for (const e of errors) console.error(`  - ${e}`);
      allOk = false;
    } else {
      console.log(`${file}: OK`);
    }
  }

  return allOk;
}

// ── Run ────────────────────────────────────────────────────────────────────

buildAll();
sortStructs();
assignFieldIds();

const header = generateHeader();
const source = generateSource();
const pythonFields = generatePythonFields();

if (process.argv.includes('--validate-presets')) {
  const ok = validatePresets();
  process.exit(ok ? 0 : 1);
} else if (process.argv.includes('--check')) {
  const existingH = readFileSync(HEADER_PATH, 'utf8');
  const existingCpp = readFileSync(SOURCE_PATH, 'utf8');
  const existingPy = existsSync(PYTHON_FIELDS_PATH) ? readFileSync(PYTHON_FIELDS_PATH, 'utf8') : '';
  let ok = true;
  if (existingH !== header) { console.error('config_parser.h is out of date'); ok = false; }
  if (existingCpp !== source) { console.error('config_parser.cpp is out of date'); ok = false; }
  if (existingPy !== pythonFields) { console.error('_gen_changeable_fields.py is out of date'); ok = false; }
  process.exit(ok ? 0 : 1);
} else if (process.argv.includes('--example')) {
  const example = generateExample();
  writeFileSync(EXAMPLE_PATH, example);
  console.log(`Generated ${EXAMPLE_PATH}`);
} else {
  writeFileSync(HEADER_PATH, header);
  writeFileSync(SOURCE_PATH, source);
  writeFileSync(PYTHON_FIELDS_PATH, pythonFields);
  console.log(`Generated ${HEADER_PATH}`);
  console.log(`Generated ${SOURCE_PATH}`);
  console.log(`Generated ${PYTHON_FIELDS_PATH}`);
  // Also validate presets if they exist
  if (existsSync(PRESETS_DIR)) {
    const ok = validatePresets();
    if (!ok) process.exit(1);
  }
}
