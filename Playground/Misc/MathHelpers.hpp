#pragma once

#include <vector>
#include <cmath>

inline float CalculateDistance(const Point& a, const Point& b) 
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static float CalculateTotalPathDistance(const std::vector<Point>& route)
{
    float totalDistance = 0.0f;
    const size_t routeSize = route.size();

#pragma omp parallel for reduction(+:totalDistance)
    for (int i = 0; i < static_cast<int>(routeSize - 1); ++i)
    {
        totalDistance += CalculateDistance(route[i], route[i + 1]);
    }

    totalDistance += CalculateDistance(route.back(), route.front());

    return totalDistance;
}

/// <summary>
/// Finds the nearest point in the list to a reference point.
/// </summary>
/// <param name="points">The list of points to search for the nearest point.</param>
/// <param name="referencePoint">The reference point to find the nearest point to.</param>
/// <returns>The nearest point in the list to the reference point.</returns>
Point FindNearestPointInList(const std::vector<Point>& points, const Point& referencePoint)
{
    float minDistance = std::numeric_limits<float>::max();
    Point nearestPoint;

    for (const auto& point : points)
    {
        float distance = CalculateDistance(referencePoint, point);

        if (distance < minDistance)
        {
            minDistance = distance;
            nearestPoint = point;
        }
    }

    return nearestPoint;
}
