// Minimal synthetic-data usage example; no external datasets are required.
#include <cdsm/paid.h>
#include <iostream>
#include <random>

int main() {
    constexpr int dimensions = 16, count = 4096, totalBudget = 1024;
    std::mt19937 random(12);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> data(count * dimensions), query(dimensions);
    for (auto& value : data) value = uniform(random);
    for (auto& value : query) value = uniform(random);
    faiss::IndexHNSWFlat index(dimensions, 16, faiss::METRIC_L2);
    index.add(count, data.data());
    const auto entries = cdsm::entryTable(index.hnsw);
    const std::vector<uint8_t> mask(count, 1); // Collect every node.
    cdsm::Workspace workspace(index); // One workspace per concurrent query.
    for (const std::string method : {"C", "F", "SCORE", "DIVERSE"}) {
        // C: continuation; F: scouts; SCORE/DIVERSE: paid entry selection.
        // The absolute cap includes primary, selection, scouts and continuation.
        const auto run = paid::search(workspace, query.data(), dimensions,
            mask, entries, method, 256, 0, 128, totalBudget);
        need(run.out.dc <= totalBudget, "Distance budget exceeded");
        std::set<int> unique;
        for (int i = 0; i < cdsm::K; ++i) {
            int id = run.out.ids[i];
            need(id >= 0 && id < count && unique.insert(id).second,
                 "Invalid or duplicate result");
            float exact = 0;
            for (int j = 0; j < dimensions; ++j) {
                float difference = query[j] - data[id * dimensions + j];
                exact += difference * difference;
            }
            need(std::abs(exact - run.out.ds[i]) < 1e-4f, "Distance mismatch");
        }
        std::cout << method << ": nearest=" << run.out.ids[0]
                  << ", distance evaluations=" << run.out.dc << '\n';
    }
}
