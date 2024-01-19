#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <omp.h>

#include "../Misc/Point.hpp"
#include "../Misc/MathHelpers.hpp"

namespace IteratedLocalSearch 
{
    /// <summary>
    /// Calculates the total distance of a given route.
    /// </summary>
    /// <param name="route">The route to calculate the distance for.</param>
    /// <returns>The total distance of the route.</returns>
    static float CalculateRouteCost(const std::vector<Point>& route)
    {
        float totalDistance = 0.0f;
        const size_t routeSize = route.size();

        for (size_t i = 0; i < routeSize - 1; ++i)
        {
            totalDistance += CalculateDistance(route[i], route[i + 1]);
        }

        totalDistance += CalculateDistance(route.back(), route.front());
        return totalDistance;
    }

    /// <summary>
    /// Perturbs a given route by shuffling its elements.
    /// </summary>
    /// <param name="route">The route to perturb.</param>
    /// <returns>The perturbed route.</returns>
    static std::vector<Point> PerturbRoute(const std::vector<Point>& route)
    {
        std::vector<Point> perturbedRoute = route;
        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(perturbedRoute.begin() + 1, perturbedRoute.end() - 1, rng);
        return perturbedRoute;
    }

    /// <summary>
    /// Performs local search to improve a given route.
    /// </summary>
    /// <param name="initialRoute">The initial route for local search.</param>
    /// <param name="start">The starting point for the route.</param>
    /// <returns>The improved route after local search.</returns>
    static std::vector<Point> LocalSearch(const std::vector<Point>& initialRoute, const Point& start)
    {
        std::vector<Point> currentRoute = initialRoute;
        float currentDistance = CalculateRouteCost(currentRoute);

        bool improvement = true;
        const size_t routeSize = currentRoute.size() - 1;

        while (improvement)
        {
            improvement = false;

#pragma omp parallel
            {
                std::vector<Point> privateRoute = currentRoute;
                float privateDistance = currentDistance;
                bool privateImprovement = false;

#pragma omp for schedule(dynamic)
                for (int i = 1; i < static_cast<int>(routeSize); ++i)
                {
                    for (size_t j = i + 1; j < routeSize; ++j)
                    {
                        std::vector<Point> newRoute = privateRoute;
                        std::reverse(newRoute.begin() + i, newRoute.begin() + j + 1);

                        float newDistance = CalculateRouteCost(newRoute);

                        if (newDistance < privateDistance)
                        {
                            privateRoute = std::move(newRoute);
                            privateDistance = newDistance;
                            privateImprovement = true;
                        }
                    }
                }

#pragma omp critical
                {
                    if (privateImprovement)
                    {
                        currentRoute = std::move(privateRoute);
                        currentDistance = privateDistance;
                        improvement = true;
                    }
                }
            }
        }

        // Ensure that the starting point is preserved
        auto it = std::find(currentRoute.begin(), currentRoute.end(), start);
        std::rotate(currentRoute.begin(), it, currentRoute.end());

        return currentRoute;
    }

    /// <summary>
    /// Finds the optimized path using Iterated Local Search (ILS).
    /// </summary>
    /// <param name="points">The list of points to visit.</param>
    /// <param name="start">The starting point for the route.</param>
    /// <param name="maxIterations">The maximum number of iterations.</param>
    /// <returns>The optimized route.</returns>
    static std::vector<Point> FindPath(const std::vector<Point>& points, const Point& start, int maxIterations = 25)
    {
        size_t n = points.size();
        Point startingPoint = FindNearestPointInList(points, start);

        std::vector<Point> bestRoute = points;
        float bestDistance = CalculateRouteCost(bestRoute);

        for (int iteration = 0; iteration < maxIterations; ++iteration)
        {
            std::vector<Point> perturbedRoute = PerturbRoute(bestRoute);
            std::vector<Point> localSearchResult = LocalSearch(perturbedRoute, startingPoint);

            float perturbedDistance = CalculateRouteCost(perturbedRoute);
            float localSearchDistance = CalculateRouteCost(localSearchResult);

            if (localSearchDistance < bestDistance)
            {
                bestRoute = std::move(localSearchResult);
                bestDistance = localSearchDistance;
            }
        }

        return bestRoute;
    }
}
