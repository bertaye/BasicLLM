#ifndef LOGGER_H
#define LOGGER_H
#include <iostream>
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    NO_LOGGING = 4
};
class Logger {
    public:
    static Logger& GetInstance() {
        static Logger instance;
        return instance;
    }

    static void LogDebug(const std::string& message) {
        if(GetInstance().logLevel <= LogLevel::DEBUG) {
            std::cout << "[DEBUG] " << message << std::endl;
        }
    }

    static void LogInfo(const std::string& message) {
        if(GetInstance().logLevel <= LogLevel::INFO) {
            std::cout << "[INFO] " << message << std::endl;
        }
    }

    static void LogWarning(const std::string& message) {
        if(GetInstance().logLevel <= LogLevel::WARNING) {
            std::cout << "[WARNING] " << message << std::endl;
        }
    }

    static void LogError(const std::string& message) {
        if(GetInstance().logLevel <= LogLevel::ERROR) {
            std::cerr << "[ERROR] " << message << std::endl;
        }
    }

    void SetLogLevel(LogLevel level) {
        logLevel = level;
    }

    LogLevel GetLogLevel() const {
        return logLevel;
    }

    private:
    Logger() = default;
    LogLevel logLevel = LogLevel::INFO; // Default log level
};

#endif //LOGGER_H