#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace AntColonyOptimization
{
    /// <summary>
    /// Initializes pheromone levels on edges with a specified initial value.
    /// </summary>
    /// <param name="pheromones">The matrix representing pheromone levels on edges.</param>
    /// <param name="initialPheromone">The initial pheromone level.</param>
    void InitializePheromones(std::vector<std::vector<float>>& pheromones, float initialPheromone)
    {
        for (int i = 0; i < pheromones.size(); ++i)
        {
            for (int j = 0; j < pheromones[i].size(); ++j)
            {
                pheromones[i][j] = initialPheromone;
            }
        }
    }

    /// <summary>
    /// Finds the index of the nearest unvisited neighbor for a given vertex.
    /// </summary>
    /// <param name="vertices">The set of vertices representing points in the problem.</param>
    /// <param name="current">The index of the current vertex.</param>
    /// <param name="visited">A boolean vector indicating visited vertices.</param>
    /// <returns>The index of the nearest unvisited neighbor.</returns>
    int FindNearestNeighbor(const std::vector<Point>& vertices, int current, const std::vector<bool>& visited)
    {
        float minDistance = std::numeric_limits<float>::max();
        int nearestIndex = -1;

        for (int i = 0; i < vertices.size(); ++i)
        {
            if (!visited[i])
            {
                float distance = CalculateDistance(vertices[current], vertices[i]);

                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearestIndex = i;
                }
            }
        }

        return nearestIndex;
    }

    /// <summary>
    /// Updates pheromone levels after an ant has completed a tour.
    /// </summary>
    /// <param name="pheromones">The matrix representing pheromone levels on edges.</param>
    /// <param name="tour">The tour completed by an ant.</param>
    /// <param name="evaporationRate">The rate at which pheromones evaporate.</param>
    /// <param name="pheromoneDeposit">The amount of pheromone deposited by the ant.</param>
    void UpdatePheromones(std::vector<std::vector<float>>& pheromones, const std::vector<Point>& tour, float evaporationRate, float pheromoneDeposit)
    {
        for (int i = 0; i < tour.size() - 1; ++i)
        {
            pheromones[i][i + 1] = (1.0 - evaporationRate) * pheromones[i][i + 1] + pheromoneDeposit;
        }

        int lastIndex = static_cast<int>(tour.size()) - 1;
        pheromones[lastIndex][0] = (1.0 - evaporationRate) * pheromones[lastIndex][0] + pheromoneDeposit;
    }

    /// <summary>
    /// Performs Ant Colony Optimization to find the best tour.
    /// </summary>
    /// <param name="vertices">The set of vertices representing points in the problem.</param>
    /// <param name="startingPosition">The starting position for the algorithm.</param>
    /// <param name="colonySize">The number of ants in the colony.</param>
    /// <param name="evaporationRate">The rate at which pheromones evaporate.</param>
    /// <param name="pheromoneDeposit">The amount of pheromone deposited by each ant.</param>
    /// <param name="maxIterations">The maximum number of iterations for the algorithm.</param>
    /// <returns>The best tour found by the Ant Colony Optimization algorithm.</returns>
    std::vector<Point> FindPath(const std::vector<Point>& vertices, const Point& startingPosition, int colonySize = 10, float evaporationRate = 0.25, float pheromoneDeposit = 1.0, int maxIterations = 120)
    {
        std::vector<Point> bestTour;
        float bestTourLength = std::numeric_limits<float>::max();

        std::vector<std::vector<float>> pheromones(vertices.size(), std::vector<float>(vertices.size(), 1.0));

#pragma omp parallel for shared(bestTourLength, bestTour, pheromones)
        for (int iteration = 0; iteration < maxIterations; ++iteration)
        {
            std::vector<std::vector<float>> localPheromones(pheromones);

#pragma omp parallel for shared(localPheromones) num_threads(colonySize)
            for (int ant = 0; ant < colonySize; ++ant)
            {
                std::vector<bool> visited(vertices.size(), false);
                std::vector<Point> tour;

                Point initialPoint = FindNearestPointInList(vertices, startingPosition);
                Point currentPoint = initialPoint;
                tour.push_back(currentPoint);
                visited[std::distance(vertices.begin(), std::find(vertices.begin(), vertices.end(), currentPoint))] = true;

                while (tour.size() < vertices.size())
                {
                    int currentIndex = std::distance(vertices.begin(), std::find(vertices.begin(), vertices.end(), currentPoint));

                    int nextIndex = FindNearestNeighbor(vertices, currentIndex, visited);

                    if (nextIndex != -1)
                    {
                        currentPoint = vertices[nextIndex];
                        tour.push_back(currentPoint);
                        visited[nextIndex] = true;
                    }
                }

                UpdatePheromones(localPheromones, tour, evaporationRate, pheromoneDeposit);

                float tourLength = std::accumulate(tour.begin(), tour.end(), 0.0,
                    [initialPoint](float sum, const Point& p) { return sum + CalculateDistance(p, initialPoint); });

#pragma omp critical
                {
                    if (tourLength < bestTourLength)
                    {
                        bestTourLength = tourLength;
                        bestTour = tour;
                    }
                }
            }

#pragma omp critical
            {
                for (int i = 0; i < pheromones.size(); ++i)
                {
                    for (int j = 0; j < pheromones[i].size(); ++j)
                    {
                        pheromones[i][j] += localPheromones[i][j];
                    }
                }
            }
        }

        for (int i = 0; i < pheromones.size(); ++i)
        {
            for (int j = 0; j < pheromones[i].size(); ++j)
            {
                pheromones[i][j] /= maxIterations;
            }
        }

        return bestTour;
    }
}
