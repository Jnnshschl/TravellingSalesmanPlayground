#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "../Misc/Point.hpp"
#include "../Misc/MathHelpers.hpp"

namespace HeldKarp 
{
    static float CalculateRouteCost(const std::vector<Point>& route)
    {
        float totalDistance = 0.0f;

        for (size_t i = 0; i < route.size() - 1; ++i)
        {
            totalDistance += CalculateDistance(route[i], route[i + 1]);
        }

        totalDistance += CalculateDistance(route.back(), route.front());
        return totalDistance;
    }

    static void GenerateSubsetsHelper(size_t size, size_t index, std::vector<size_t>& subset, std::vector<std::vector<size_t>>& subsets)
    {
        if (subset.size() == size)
        {
            subsets.push_back(subset);
            return;
        }

        for (size_t i = index; i < size; ++i)
        {
            subset.push_back(i);
            GenerateSubsetsHelper(size, i + 1, subset, subsets);
            subset.pop_back();
        }
    }

    static std::vector<std::vector<size_t>> GenerateSubsets(size_t size)
    {
        std::vector<std::vector<size_t>> subsets;
        std::vector<size_t> subset;
        GenerateSubsetsHelper(size, 0, subset, subsets);
        return subsets;
    }

    static std::vector<Point> FindPath(const std::vector<Point>& points, const Point& start)
    {
        size_t n = points.size();

        // Generate all subsets of cities
        auto subsets = GenerateSubsets(n);

        // Initialize the memoization table
        std::vector<std::vector<float>> memo(n, std::vector<float>(1 << n, std::numeric_limits<float>::infinity()));

        // Find the nearest point to the starting point in the list
        Point startingPoint = FindNearestPointInList(points, start);

        // Set up initial distances from startingPoint to other cities
        for (size_t i = 0; i < n; ++i)
        {
            memo[i][1 << i] = CalculateDistance(startingPoint, points[i]);
        }

        // Dynamic programming step
        for (size_t subsetSize = 2; subsetSize <= n; ++subsetSize)
        {
            for (const auto& subset : subsets)
            {
                if (subset.size() != subsetSize || subset[0] != 0)
                {
                    continue;
                }

                for (size_t endCity : subset)
                {
                    size_t subsetWithoutEnd = 0;

                    for (size_t city : subset)
                    {
                        if (city != endCity)
                        {
                            subsetWithoutEnd |= (1 << city);
                        }
                    }

                    for (size_t startCity : subset)
                    {
                        if (startCity != endCity)
                        {
                            float distance = CalculateDistance(points[startCity], points[endCity]);
                            memo[endCity][subsetWithoutEnd] = std::min(memo[endCity][subsetWithoutEnd], memo[startCity][subsetWithoutEnd] + distance);
                        }
                    }
                }
            }
        }

        // Reconstruct the optimal route
        size_t fullSubset = (1 << n) - 1;
        size_t currentCity = 0;
        size_t currentSubset = fullSubset;

        std::vector<size_t> optimalRouteIndices;

        for (size_t i = 0; i < n; ++i)
        {
            optimalRouteIndices.push_back(currentCity);
            size_t nextCity = 0;

            for (size_t city = 1; city < n; ++city)
            {
                if ((currentSubset >> city) & 1)
                {
                    if (memo[currentCity][currentSubset] == memo[city][currentSubset ^ (1 << currentCity)] + CalculateDistance(points[city], points[currentCity]))
                    {
                        nextCity = city;
                        break;
                    }
                }
            }

            currentSubset ^= (1 << currentCity);
            currentCity = nextCity;
        }

        // Convert indices to actual points to get the optimal route
        std::vector<Point> optimalRoute;

        for (size_t index : optimalRouteIndices)
        {
            optimalRoute.push_back(points[index]);
        }

        return optimalRoute;
    }
}
