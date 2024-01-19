#pragma once

#include <cmath>
#include <random>
#include <vector>

#include <omp.h>

#include "../Misc/Point.hpp"
#include "../Misc/MathHelpers.hpp"

namespace SimulatedAnnealing
{
    /// <summary>
    /// Swaps two cities in the given route.
    /// </summary>
    /// <param name="route">The route to perform the swap on.</param>
    /// <param name="rng">Random number generator.</param>
    /// <returns>The route after swapping two cities.</returns>
    static std::vector<Point> swapCities(const std::vector<Point>& route, std::mt19937& rng)
    {
        std::vector<Point> newRoute = route;
        std::uniform_int_distribution<size_t> dist(0, route.size() - 1);
        size_t index1 = dist(rng);
        size_t index2 = dist(rng);

        while (index2 == index1)
        {
            index2 = dist(rng);
        }

        std::swap(newRoute[index1], newRoute[index2]);
        return newRoute;
    }

    /// <summary>
    /// Finds the optimized path using Simulated Annealing algorithm.
    /// </summary>
    /// <param name="initialRoute">The initial route for the algorithm.</param>
    /// <param name="startingPosition">The starting position for the route.</param>
    /// <param name="initialTemperature">The initial temperature for the annealing process.</param>
    /// <param name="coolingRate">The cooling rate for the temperature.</param>
    /// <param name="iterations">The number of iterations for the algorithm.</param>
    /// <returns>The optimized route.</returns>
    static std::vector<Point> FindPath(const std::vector<Point>& initialRoute, const Point& startingPosition, float initialTemperature = 100.0f, float coolingRate = 0.995, int iterations = 10000)
    {
        std::vector<Point> currentRoute = initialRoute;
        float currentDistance = CalculateTotalPathDistance(currentRoute);

        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        float temperature = initialTemperature;

#pragma omp parallel
        {
            std::vector<Point> localCurrentRoute = currentRoute;
            float localCurrentDistance = currentDistance;

#pragma omp for schedule(dynamic) nowait
            for (int iteration = 0; iteration < iterations; ++iteration)
            {
                std::vector<Point> newRoute = swapCities(localCurrentRoute, rng);
                float newDistance = CalculateTotalPathDistance(newRoute);

                float delta = newDistance - localCurrentDistance;

                if (delta < 0 || (dist(rng) < std::exp(-delta / temperature)))
                {
                    localCurrentRoute = newRoute;
                    localCurrentDistance = newDistance;
                }
                else if (dist(rng) < std::exp(-delta / (0.1 * temperature)))
                {
                    localCurrentRoute = swapCities(localCurrentRoute, rng);
                    localCurrentDistance = CalculateTotalPathDistance(localCurrentRoute);
                }
            }

#pragma omp critical
            {
                if (localCurrentDistance < currentDistance)
                {
                    currentRoute = localCurrentRoute;
                    currentDistance = localCurrentDistance;
                }
            }

#pragma omp barrier
#pragma omp single
            {
                temperature *= coolingRate;
            }
        }

        return currentRoute;
    }
}
