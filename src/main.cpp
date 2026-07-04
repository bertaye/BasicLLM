#include <cstdio>

#include "Tensor.h"
#include "TensorOps.h"
#include "Model.h"
#include <thread>

int main() {
    Logger::GetInstance().SetLogLevel(LogLevel::INFO);
    std::filesystem::path stroies15M_path = std::filesystem::current_path() / ".." / "models" / "stories15M.bin";
    std::filesystem::path tokenizer_path = std::filesystem::current_path() / ".." / "models" / "tokenizer.bin";
    Model model(stroies15M_path.string(), tokenizer_path.string());
    std::string prompt = "Once upon a time";
    model.enableLiveMetrics();
    model.Generate(prompt, 20, 0.0f, std::cout, true);
    std::cout << std::endl;
    std::cout << "Performance Metrics:" << std::endl;
    std::cout << "Tokens per sec: " << (double)model.getMetrics()->generatedTokens.load() / model.getMetrics()->generateSeconds.load() << std::endl;
    model.enableLiveMetrics();
    model.GenerateWithKVCache(prompt, 20, 0.0f, std::cout, true);
    std::cout << std::endl;
    std::cout << "Performance Metrics for KV cache:" << std::endl;
    std::cout << "Tokens per sec: " << (double)model.getMetrics()->generatedTokens.load() / model.getMetrics()->generateSeconds.load() << std::endl;
    return 0;
}
