#ifndef BINARY_FILE_LOADER_H
#define BINARY_FILE_LOADER_H
//A simple header file for loading llm weights & tokenizer data
#include <filesystem>
#include <memory>
#include <fstream>
class BinaryFileLoader{
    public:
    inline static std::unique_ptr<char[]> loadFile(std::filesystem::path path)
    {
        if(!std::filesystem::exists(path))
        {
            return nullptr;
        }

        auto fileSize = std::filesystem::file_size(path);
        std::ifstream file(path, std::ios::binary);
        auto buffer = std::make_unique<char[]>(fileSize);
        file.read(buffer.get(), fileSize);
        return buffer;
    }
};

#endif //BINARY_FILE_LOADER_H