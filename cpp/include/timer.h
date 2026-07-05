#pragma once

#include<chrono>
#include<string>
#include<unordered_map>

class Timerecorder{
    public:
        void add(const std::string &name, double cost);
        double get(const std::string &name) const;
        std::unordered_map<std::string , double> summary() const;

    private:
        std::unordered_map<std::string, double> recorders;
};

class scopedTimer{
    public:
        scopedTimer(Timerecorder &recorder, const std::string &name);
        ~scopedTimer();

        scopedTimer(const scopedTimer&) = delete;
        scopedTimer &operator = (const scopedTimer&) = delete;

    private:
        using Clock = std::chrono::steady_clock;

        Timerecorder &recorder_;
        std::string name_;
        Clock::time_point start;

};