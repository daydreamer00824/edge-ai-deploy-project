#include <string>
#include <chrono>
#include <unordered_map>
#include "timer.h"

void Timerecorder::add(const std::string &name, double cost){
    recorders[name] +=cost;
}

double Timerecorder::get(const std::string &name) const{
    auto it = recorders.find(name);
    if(it == recorders.end()){
        return 0.0;
    }

    return it -> second;
}

std::unordered_map<std::string , double> Timerecorder::summary() const{
    return recorders;
}

scopedTimer::scopedTimer(Timerecorder &recorder, const std::string &name):
    recorder_(recorder),
    name_(name),
    start(Clock::now()){}

scopedTimer::~scopedTimer(){
    auto end = Clock::now();

    double cost = std::chrono::duration<double, std::milli>(end - start).count();
    recorder_.add(name_, cost);
}