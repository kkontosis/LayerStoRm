#!/usr/bin/env bash
# Validate config/schema.json and test configs against it.
# Requires: pip install jsonschema
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCHEMA="$PROJECT_DIR/config/schema.json"
VALID_CONFIG="$PROJECT_DIR/test-data/config/valid_deepseek_v3_2.json"

PASS=0
FAIL=0

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=== LayerStoRm Schema Validation ==="
echo

# 1. Check schema is valid JSON
echo "[1] Schema is valid JSON"
if python3 -m json.tool "$SCHEMA" > /dev/null 2>&1; then
  pass "schema.json parses as valid JSON"
else
  fail "schema.json is not valid JSON"
fi

# 2. Check schema is valid JSON Schema 2020-12
echo "[2] Schema is valid JSON Schema 2020-12"
if python3 -c "
import json, jsonschema
with open('$SCHEMA') as f:
    schema = json.load(f)
jsonschema.Draft202012Validator.check_schema(schema)
" 2>&1; then
  pass "schema.json is valid JSON Schema 2020-12"
else
  fail "schema.json is not valid JSON Schema 2020-12"
fi

# 3. Validate reference config against schema
echo "[3] Reference config validates against schema"
if python3 -c "
import json, jsonschema
with open('$SCHEMA') as f:
    schema = json.load(f)
with open('$VALID_CONFIG') as f:
    config = json.load(f)
jsonschema.validate(config, schema, cls=jsonschema.Draft202012Validator)
" 2>&1; then
  pass "valid_deepseek_v3_2.json validates"
else
  fail "valid_deepseek_v3_2.json does not validate"
fi

# 4. Negative: missing required field (model.architecture)
echo "[4] Reject config missing required field"
if python3 -c "
import json, jsonschema
with open('$SCHEMA') as f:
    schema = json.load(f)
with open('$VALID_CONFIG') as f:
    config = json.load(f)
del config['model']['architecture']
try:
    jsonschema.validate(config, schema, cls=jsonschema.Draft202012Validator)
    raise SystemExit(1)
except jsonschema.ValidationError:
    pass
" 2>&1; then
  pass "rejects missing model.architecture"
else
  fail "accepted config without model.architecture"
fi

# 5. Negative: wrong type (model.num_hidden_layers as string)
echo "[5] Reject wrong type"
if python3 -c "
import json, jsonschema
with open('$SCHEMA') as f:
    schema = json.load(f)
with open('$VALID_CONFIG') as f:
    config = json.load(f)
config['model']['num_hidden_layers'] = 'sixty-one'
try:
    jsonschema.validate(config, schema, cls=jsonschema.Draft202012Validator)
    raise SystemExit(1)
except jsonschema.ValidationError:
    pass
" 2>&1; then
  pass "rejects string for model.num_hidden_layers"
else
  fail "accepted string for model.num_hidden_layers"
fi

# 6. Negative: out-of-range value
echo "[6] Reject out-of-range value"
if python3 -c "
import json, jsonschema
with open('$SCHEMA') as f:
    schema = json.load(f)
with open('$VALID_CONFIG') as f:
    config = json.load(f)
config['model']['num_hidden_layers'] = 0
try:
    jsonschema.validate(config, schema, cls=jsonschema.Draft202012Validator)
    raise SystemExit(1)
except jsonschema.ValidationError:
    pass
" 2>&1; then
  pass "rejects num_hidden_layers=0"
else
  fail "accepted num_hidden_layers=0"
fi

# 7. Negative: invalid enum value
echo "[7] Reject invalid enum value"
if python3 -c "
import json, jsonschema
with open('$SCHEMA') as f:
    schema = json.load(f)
with open('$VALID_CONFIG') as f:
    config = json.load(f)
config['model']['architecture'] = 'llama'
try:
    jsonschema.validate(config, schema, cls=jsonschema.Draft202012Validator)
    raise SystemExit(1)
except jsonschema.ValidationError:
    pass
" 2>&1; then
  pass "rejects architecture='llama'"
else
  fail "accepted architecture='llama'"
fi

# 8. Negative: additional property rejected
echo "[8] Reject additional properties"
if python3 -c "
import json, jsonschema
with open('$SCHEMA') as f:
    schema = json.load(f)
with open('$VALID_CONFIG') as f:
    config = json.load(f)
config['model']['bogus_field'] = 42
try:
    jsonschema.validate(config, schema, cls=jsonschema.Draft202012Validator)
    raise SystemExit(1)
except jsonschema.ValidationError:
    pass
" 2>&1; then
  pass "rejects unknown model.bogus_field"
else
  fail "accepted unknown model.bogus_field"
fi

# 9. Negative: capture_expert_ffn must be false (INV-0.6)
echo "[9] Reject capture_expert_ffn=true (INV-0.6)"
if python3 -c "
import json, jsonschema
with open('$SCHEMA') as f:
    schema = json.load(f)
with open('$VALID_CONFIG') as f:
    config = json.load(f)
config['compute']['cuda_graphs']['capture_expert_ffn'] = True
try:
    jsonschema.validate(config, schema, cls=jsonschema.Draft202012Validator)
    raise SystemExit(1)
except jsonschema.ValidationError:
    pass
" 2>&1; then
  pass "rejects capture_expert_ffn=true"
else
  fail "accepted capture_expert_ffn=true"
fi

# 10. Positive: minimal config (only required sections/fields)
echo "[10] Accept minimal config"
if python3 -c "
import json, jsonschema
with open('$SCHEMA') as f:
    schema = json.load(f)
minimal = {
    'model': {
        'architecture': 'deepseek_v3',
        'weights_path': '/data/models/test/',
        'weights_format': 'safetensors',
        'num_hidden_layers': 61,
        'hidden_size': 7168,
        'num_attention_heads': 128,
        'num_key_value_heads': 128,
        'intermediate_size': 18432,
        'n_routed_experts': 256,
        'num_experts_per_tok': 8,
        'vocab_size': 129280,
        'max_position_embeddings': 163840
    },
    'quantization': {
        'weights': 'nvfp4',
        'attention_compute': 'fp8_e4m3',
        'kv_cache': 'fp8_e4m3',
        'gating_compute': 'fp16'
    },
    'hardware': {
        'gpus': [{'id': 0, 'type': 'rtx5090', 'vram_gb': 32}],
        'system_ram_gb': 128
    }
}
jsonschema.validate(minimal, schema, cls=jsonschema.Draft202012Validator)
" 2>&1; then
  pass "minimal config validates"
else
  fail "minimal config rejected"
fi

echo
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
