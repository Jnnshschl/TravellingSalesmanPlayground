#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include <SDL.h>

#include "Misc/Polygon.hpp"
#include "Pathfinding/AntColonyOptimization.hpp"
#include "Pathfinding/GeneticAlgorithm.hpp"
#include "Pathfinding/HeldKarp.hpp"
#include "Pathfinding/IteratedLocalSearch.hpp"
#include "Pathfinding/NearestNeighbor.hpp"
#include "Pathfinding/ParticleSwarmOptimization.hpp"
#include "Pathfinding/SimulatedAnnealing.hpp"

template <typename F>
inline long long measureExecutionTime(F func) 
{
    auto start_time = std::chrono::high_resolution_clock::now();
    func();
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
}

static void Render(SDL_Window* window, SDL_Renderer* renderer, const Point& mousePos, const Polygon& poly, bool lclick, bool rclick, bool mclick) noexcept;
