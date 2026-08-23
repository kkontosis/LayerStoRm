#pragma once
// Unified end-of-run x-ray report: a single table mixing COARSE token-breakdown
// rows with INDENTED finer sub-rows, followed by a free-form stats block, and a
// machine-readable JSON sidecar so results can be rendered as tables
// programmatically (and fed to tools/loader_xray). CUDA-free; pure formatting.
//
// The caller builds a Report (coarse rows from its own per-command timers, fine
// rows from perf_trace::aggregate(), stats from whatever it measured) and calls
// emit() once. Columns are auto-blanked when a value is negative.
#include <string>
#include <utility>
#include <vector>

namespace layerstorm::perf_report {

struct Row {
    std::string name;
    int    indent       = 0;     // 0 = coarse; 1/2 = nested fine sub-rows
    double ms_per_token = 0.0;
    double pct          = -1.0;  // % of token wall; < 0 → blank cell
    double ms_per_op    = -1.0;  // < 0 → blank cell
    double n_per_token  = -1.0;  // < 0 → blank cell
    bool   section      = false; // true → render as a "── name ──" divider, no columns
};

// A generic multi-column table (e.g. per-GPU VRAM breakdown). headers[0] labels
// the row-name column (left-aligned); the rest are right-aligned value columns.
struct Grid {
    std::string title;
    std::vector<std::string> headers;            // headers[0] = row-label column
    std::vector<std::vector<std::string>> rows;  // each row: cells aligned to headers
    void add(std::vector<std::string> cells) { rows.push_back(std::move(cells)); }
};

struct Report {
    std::string title;
    std::vector<Row> rows;
    std::vector<std::pair<std::string, std::string>> stats;  // ordered key → value
    std::vector<Grid> grids;                                 // extra tables (rendered after stats)

    void row(std::string name, int indent, double mspt, double pct = -1.0,
             double mso = -1.0, double npt = -1.0) {
        rows.push_back({std::move(name), indent, mspt, pct, mso, npt, false});
    }
    void section(std::string name) {
        rows.push_back({std::move(name), 0, 0, -1, -1, -1, true});
    }
    void stat(std::string key, std::string value) {
        stats.emplace_back(std::move(key), std::move(value));
    }
    Grid& grid(std::string title, std::vector<std::string> headers) {
        grids.push_back({std::move(title), std::move(headers), {}});
        return grids.back();
    }
};

// Aligned fixed-width ASCII table (coarse + indented fine) + a "── stats ──" block.
std::string render_table(const Report& r);
// {title, rows:[{name,indent,ms_per_token,pct,ms_per_op,n_per_token,section}], stats:{..}}.
std::string render_json(const Report& r);
// Print render_table() to stderr; if json_path is non-empty also write render_json() there.
void emit(const Report& r, const char* json_path);

}  // namespace layerstorm::perf_report
