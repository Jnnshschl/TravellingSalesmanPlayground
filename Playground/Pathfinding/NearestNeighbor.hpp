#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

namespace NearestNeighbor
{
    /// <summary>
    /// Finds the path using the Nearest Neighbor algorithm starting from a specified point.
    /// </summary>
    /// <param name="vertices">The set of points representing the vertices of the problem.</param>
    /// <param name="start">The starting point for the algorithm.</param>
    /// <returns>The path found using the Nearest Neighbor algorithm.</returns>
    static std::vector<Point> FindPath(const std::vector<Point>& vertices, const Point& start)
    {
        std::vector<Point> route;
        std::vector<bool> visited(vertices.size(), false);

        Point nearestStart = FindNearestPointInList(vertices, start);
        route.push_back(nearestStart);

        auto startIt = std::find(vertices.begin(), vertices.end(), nearestStart);

        if (startIt != vertices.end())
        {
            visited[std::distance(vertices.begin(), startIt)] = true;
        }

        while (route.size() < vertices.size())
        {
            float minDistance = std::numeric_limits<float>::max();
            int nearestIndex = -1;

            for (int i = 0; i < vertices.size(); ++i)
            {
                if (!visited[i])
                {
                    float distance = CalculateDistance(route.back(), vertices[i]);

                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        nearestIndex = i;
                    }
                }
            }

            if (nearestIndex != -1)
            {
                route.push_back(vertices[nearestIndex]);
                visited[nearestIndex] = true;
            }
        }

        return route;
    }
}
