#include "core/perf_report.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace layerstorm::perf_report {
namespace {

// Right-aligned numeric cell of width w, or a centered "-" when v < 0 (blank).
std::string cell(double v, int w, int prec, const char* suffix = "") {
    std::ostringstream t;
    if (v < 0.0) {
        t << "-";
    } else {
        t << std::fixed << std::setprecision(prec) << v << suffix;
    }
    std::ostringstream o;
    o << std::setw(w) << t.str();
    return o.str();
}

void json_num(std::ostringstream& o, double v) {
    if (v < 0.0) o << "null";
    else         o << std::fixed << std::setprecision(4) << v;
}

}  // namespace

std::string render_table(const Report& r) {
    constexpr int kName = 26, kMs = 9, kPct = 8, kOp = 10, kN = 9;
    std::ostringstream o;
    o << "\n── " << r.title << " (avg per token) ──\n";
    o << std::left << std::setw(kName) << "stage" << std::right
      << std::setw(kMs) << "ms/tok" << std::setw(kPct) << "%tok"
      << std::setw(kOp) << "ms/op" << std::setw(kN) << "n/tok" << "\n";

    for (const Row& row : r.rows) {
        if (row.section) {
            o << "── " << row.name << " ──\n";
            continue;
        }
        std::string name(static_cast<size_t>(row.indent) * 2, ' ');
        name += row.name;
        o << std::left << std::setw(kName) << name << std::right
          << cell(row.ms_per_token, kMs, 3)
          << cell(row.pct, kPct, 1, "%")
          << cell(row.ms_per_op, kOp, 4)
          << cell(row.n_per_token, kN, 1) << "\n";
    }

    if (!r.stats.empty()) {
        o << "── stats ──\n";
        size_t w = 0;
        for (const auto& s : r.stats) w = std::max(w, s.first.size());
        for (const auto& s : r.stats)
            o << "  " << std::left << std::setw(static_cast<int>(w) + 2) << s.first
              << s.second << "\n";
    }

    for (const Grid& g : r.grids) {
        const size_t ncol = g.headers.size();
        std::vector<size_t> w(ncol, 0);
        for (size_t c = 0; c < ncol; ++c) w[c] = g.headers[c].size();
        for (const auto& row : g.rows)
            for (size_t c = 0; c < ncol && c < row.size(); ++c)
                w[c] = std::max(w[c], row[c].size());
        o << "── " << g.title << " ──\n";
        auto emit_row = [&](const std::vector<std::string>& cells) {
            for (size_t c = 0; c < ncol; ++c) {
                const std::string& s = c < cells.size() ? cells[c] : std::string();
                if (c == 0) o << std::left << std::setw(static_cast<int>(w[c]) + 2) << s;
                else        o << std::right << std::setw(static_cast<int>(w[c]) + 2) << s;
            }
            o << "\n";
        };
        emit_row(g.headers);
        for (const auto& row : g.rows) emit_row(row);
    }
    return o.str();
}

std::string render_json(const Report& r) {
    std::ostringstream o;
    o << "{\n  \"title\": \"" << r.title << "\",\n  \"rows\": [\n";
    for (size_t i = 0; i < r.rows.size(); ++i) {
        const Row& row = r.rows[i];
        o << "    {\"name\": \"" << row.name << "\", \"indent\": " << row.indent
          << ", \"section\": " << (row.section ? "true" : "false")
          << ", \"ms_per_token\": "; json_num(o, row.section ? -1.0 : row.ms_per_token);
        o << ", \"pct\": ";          json_num(o, row.pct);
        o << ", \"ms_per_op\": ";    json_num(o, row.ms_per_op);
        o << ", \"n_per_token\": ";  json_num(o, row.n_per_token);
        o << "}" << (i + 1 < r.rows.size() ? "," : "") << "\n";
    }
    o << "  ],\n  \"stats\": {\n";
    for (size_t i = 0; i < r.stats.size(); ++i)
        o << "    \"" << r.stats[i].first << "\": \"" << r.stats[i].second << "\""
          << (i + 1 < r.stats.size() ? "," : "") << "\n";
    o << "  },\n  \"grids\": [\n";
    for (size_t gi = 0; gi < r.grids.size(); ++gi) {
        const Grid& g = r.grids[gi];
        o << "    {\"title\": \"" << g.title << "\", \"headers\": [";
        for (size_t c = 0; c < g.headers.size(); ++c)
            o << "\"" << g.headers[c] << "\"" << (c + 1 < g.headers.size() ? ", " : "");
        o << "], \"rows\": [";
        for (size_t ri = 0; ri < g.rows.size(); ++ri) {
            o << "[";
            for (size_t c = 0; c < g.rows[ri].size(); ++c)
                o << "\"" << g.rows[ri][c] << "\"" << (c + 1 < g.rows[ri].size() ? ", " : "");
            o << "]" << (ri + 1 < g.rows.size() ? ", " : "");
        }
        o << "]}" << (gi + 1 < r.grids.size() ? "," : "") << "\n";
    }
    o << "  ]\n}\n";
    return o.str();
}

void emit(const Report& r, const char* json_path) {
    std::cerr << render_table(r);
    if (json_path && *json_path) {
        std::ofstream f(json_path);
        if (f) {
            f << render_json(r);
            std::cerr << "  [x-ray json → " << json_path << "]\n";
        } else {
            std::cerr << "  [x-ray json: FAILED to open " << json_path << "]\n";
        }
    }
}

}  // namespace layerstorm::perf_report
