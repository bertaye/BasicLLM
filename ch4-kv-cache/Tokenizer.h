#ifndef TOKENIZER_H
#define TOKENIZER_H
#include "BinaryFileLoader.h"
#include <string>
#include <map>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>

// Byte-Pair Encoding tokenizer for the llama2.c `tokenizer.bin` format.
// The binary layout (written by llama2.c's tokenizer.py) is:
//   [int32]              maxTokenLength           - longest token in bytes
//   repeated vocabSize times, one record per token id:
//     [float32]          score                    - BPE merge score
//     [int32]            length                   - token byte length
//     [byte * length]    bytes                    - token text (not null-terminated)
// Note: vocabSize is NOT stored in the file; it comes from the model config.
class Tokenizer{
    public:
    // The first three vocab entries are the special tokens <unk>, <s>, </s>.
    static constexpr int BosToken = 1; // <s>  beginning of sequence
    static constexpr int EosToken = 2; // </s> end of sequence

    struct TokenizerConfig{
        int vocabSize = 0;
        std::vector<std::string> vocab;
        std::vector<float> vocabScores;
        std::map<std::string, int> sortedVocab; // token text -> id, for fast lookup
        unsigned int maxTokenLength = 0;
    };
    Tokenizer() = default;
    // Parse the tokenizer.bin buffer into config. vocabSize comes from the model.
    Tokenizer(std::filesystem::path filePath, int vocabSize)
    {
        auto tokenizerBin = BinaryFileLoader::loadFile(filePath);
        if(!tokenizerBin)
        {
            throw std::runtime_error("Failed to load tokenizer binary file.");
        }

        config.vocabSize = vocabSize;
        config.vocab.resize(vocabSize);
        config.vocabScores.resize(vocabSize);

        const char* cursor = tokenizerBin.get();

        std::memcpy(&config.maxTokenLength, cursor, sizeof(unsigned int));
        cursor += sizeof(unsigned int);

        for(int i = 0; i < vocabSize; ++i)
        {
            float score;
            std::memcpy(&score, cursor, sizeof(float));
            cursor += sizeof(float);
            config.vocabScores[i] = score;

            int length;
            std::memcpy(&length, cursor, sizeof(int));
            cursor += sizeof(int);

            config.vocab[i] = std::string(cursor, length);
            cursor += length;

            // emplace keeps the first (lowest) id if a string ever repeats.
            config.sortedVocab.emplace(config.vocab[i], i);
        }
    }

    Tokenizer(const Tokenizer& other)
    {
        this->config = other.config;
    }

    Tokenizer(Tokenizer&& other) noexcept
    {
        this->config = std::move(other.config);
    }

    Tokenizer& operator=(const Tokenizer& other)
    {
        if (this != &other)
        {
            this->config = other.config;
        }
        return *this;
    }

    Tokenizer& operator=(Tokenizer&& other) noexcept
    {
        if (this != &other)
        {
            this->config = std::move(other.config);
        }
        return *this;
    }

    int VocabSize() const { return config.vocabSize; }

    // Convert a single token id back into its text piece.
    // prevToken is needed to strip the leading space sentencepiece adds after BOS.
    std::string Decode(int prevToken, int token) const
    {
        const std::string& piece = config.vocab[token];
        const char* start = piece.c_str();

        // sentencepiece prepends a space to the first real token; drop it after BOS.
        if(prevToken == BosToken && start[0] == ' ')
        {
            start++;
        }

        // Raw byte tokens are stored as the literal text "<0xNN>"; emit the byte itself.
        unsigned int byteValue;
        if(std::sscanf(start, "<0x%02X>", &byteValue) == 1)
        {
            return std::string(1, static_cast<char>(byteValue));
        }

        return std::string(start);
    }

    // Encode UTF-8 text into token ids using the BPE merge algorithm.
    std::vector<int> Encode(const std::string& text, bool addBos, bool addEos) const
    {
        // BPE start
        std::vector<int> tokens;

        if(addBos)
        {
            tokens.push_back(BosToken);
        }

        // sentencepiece prepends a dummy space token when the input is non-empty.
        if(!text.empty())
        {
            auto it = config.sortedVocab.find(" ");
            if(it != config.sortedVocab.end())
            {
                tokens.push_back(it->second);
            }
        }

        // First pass: walk UTF-8 characters. Look each one up in the vocab; if it
        // isn't there, fall back to raw byte tokens (byteValue + 3, skipping the
        // three special tokens that occupy ids 0..2).
        std::string current;
        for(size_t i = 0; i < text.size(); ++i)
        {
            unsigned char byte = static_cast<unsigned char>(text[i]);

            // A UTF-8 continuation byte matches 0b10xxxxxx; anything else starts a
            // new codepoint, so reset the accumulator.
            if((byte & 0xC0) != 0x80)
            {
                current.clear();
            }
            current.push_back(static_cast<char>(byte));

            // If the next byte continues this codepoint (and we have room), keep going.
            bool nextIsContinuation = (i + 1 < text.size()) &&
                (static_cast<unsigned char>(text[i + 1]) & 0xC0) == 0x80;
            if(nextIsContinuation && current.size() < 4)
            {
                continue;
            }

            auto it = config.sortedVocab.find(current);
            if(it != config.sortedVocab.end())
            {
                tokens.push_back(it->second);
            }
            else
            {
                for(unsigned char fallbackByte : current)
                {
                    tokens.push_back(static_cast<int>(fallbackByte) + 3);
                }
            }
            current.clear();
        }

        // Merge pass: repeatedly merge the adjacent pair whose concatenation exists
        // in the vocab with the highest score, until no more merges are possible.
        while(true)
        {
            float bestScore = -1e10f;
            int bestId = -1;
            int bestIndex = -1;

            for(size_t i = 0; i + 1 < tokens.size(); ++i)
            {
                std::string merged = config.vocab[tokens[i]] + config.vocab[tokens[i + 1]];
                auto it = config.sortedVocab.find(merged);
                if(it != config.sortedVocab.end() && config.vocabScores[it->second] > bestScore)
                {
                    bestScore = config.vocabScores[it->second];
                    bestId = it->second;
                    bestIndex = static_cast<int>(i);
                }
            }

            if(bestIndex == -1)
            {
                break; // no adjacent pair can be merged
            }

            // Merge the pair: replace the left token, drop the right one.
            tokens[bestIndex] = bestId;
            tokens.erase(tokens.begin() + bestIndex + 1);
        }

        if(addEos)
        {
            tokens.push_back(EosToken);
        }
        // BPE end

        return tokens;
    }

    private:
    TokenizerConfig config;
};

#endif //TOKENIZER_H
