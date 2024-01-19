#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <omp.h>

namespace GeneticAlgorithm 
{
    /// <summary>
    /// Performs local search to improve a given route.
    /// </summary>
    /// <param name="route">The route to apply local search to.</param>
    static void LocalSearch(std::vector<Point>& route)
    {
        bool improvement = true;
        const size_t routeSize = route.size();
        const size_t routeSizeMinusTwo = routeSize - 2;

        while (improvement)
        {
            improvement = false;

#pragma omp parallel for
            for (int i = 1; i < static_cast<int>(routeSizeMinusTwo); ++i)
            {
                for (size_t j = static_cast<size_t>(i + 1); j < routeSizeMinusTwo; ++j)
                {
                    float originalDistance = CalculateDistance(route[i - 1], route[i]) + CalculateDistance(route[j], route[j + 1]);
                    float newDistance = CalculateDistance(route[i - 1], route[j]) + CalculateDistance(route[i], route[j + 1]);
                    if (newDistance < originalDistance)
                    {
#pragma omp critical
                        {
                            std::reverse(route.begin() + i, route.begin() + j + 1);
                            improvement = true;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Performs crossover operation on two parent routes to create a child route.
    /// </summary>
    /// <param name="parent1">The first parent route.</param>
    /// <param name="parent2">The second parent route.</param>
    /// <param name="rng">Random number generator.</param>
    /// <returns>The resulting child route after crossover.</returns>
    static std::vector<Point> Crossover(const std::vector<Point>& parent1, const std::vector<Point>& parent2, std::mt19937& rng)
    {
        std::vector<Point> child = parent1;

        size_t start = std::uniform_int_distribution<size_t>(0, parent1.size() - 2)(rng);
        size_t end = std::uniform_int_distribution<size_t>(start + 1, parent1.size() - 1)(rng);

        if (start > end)
        {
            std::swap(start, end);
        }

        std::vector<bool> selected(parent1.size(), false);

        for (size_t i = start; i <= end; ++i)
        {
            child[i] = parent2[i];
            selected[i] = true;
        }

        size_t index = (end + 1) % parent1.size();

        for (size_t i = (end + 1) % parent1.size(); i != start; i = (i + 1) % parent1.size())
        {
            while (selected[index])
            {
                index = (index + 1) % parent1.size();
            }

            child[i] = parent2[index];
            index = (index + 1) % parent1.size();
        }

        return child;
    }

    /// <summary>
    /// Performs mutation operation on a given route based on a mutation rate.
    /// </summary>
    /// <param name="route">The route to mutate.</param>
    /// <param name="mutationRate">The probability of mutation for each gene.</param>
    /// <param name="rng">Random number generator.</param>
    static void Mutate(std::vector<Point>& route, float mutationRate, std::mt19937& rng)
    {
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

        const size_t routeSizeMinusOne = route.size() - 1;

#pragma omp parallel for
        for (int i = 1; i < static_cast<int>(routeSizeMinusOne); ++i)
        {
            if (distribution(rng) < mutationRate)
            {
                size_t j = std::uniform_int_distribution<size_t>(1, routeSizeMinusOne)(rng);
                std::swap(route[i], route[j]);
            }
        }
    }

    /// <summary>
    /// Performs tournament selection to choose two parent routes.
    /// </summary>
    /// <param name="population">The population of routes to choose from.</param>
    /// <param name="tournamentSize">The size of the tournament.</param>
    /// <param name="rng">Random number generator.</param>
    /// <returns>A pair of selected parent routes.</returns>
    static std::pair<std::vector<Point>, std::vector<Point>> TournamentSelection(const std::vector<std::vector<Point>>& population, size_t tournamentSize, std::mt19937& rng)
    {
        std::vector<size_t> indices(population.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::shuffle(indices.begin(), indices.end(), rng);

        size_t bestIndex1 = indices[0];

        for (size_t i = 1; i < tournamentSize; ++i)
        {
            if (CalculateTotalPathDistance(population[indices[i]]) < CalculateTotalPathDistance(population[bestIndex1]))
            {
                bestIndex1 = indices[i];
            }
        }

        size_t bestIndex2 = indices[1];

        for (size_t i = 2; i < tournamentSize; ++i)
        {
            if (CalculateTotalPathDistance(population[indices[i]]) < CalculateTotalPathDistance(population[bestIndex2]))
            {
                bestIndex2 = indices[i];
            }
        }

        return { population[bestIndex1], population[bestIndex2] };
    }

    /// <summary>
    /// Initializes the population with random routes, ensuring a fixed starting point.
    /// </summary>
    /// <param name="vertices">The set of points representing the vertices of the problem.</param>
    /// <param name="populationSize">The size of the population.</param>
    /// <param name="rng">Random number generator.</param>
    /// <param name="fixedPoint">The fixed starting point for all routes.</param>
    /// <returns>The initialized population of routes.</returns>
    static std::vector<std::vector<Point>> InitializePopulation(const std::vector<Point>& vertices, size_t populationSize, std::mt19937& rng, const Point& fixedPoint)
    {
        std::vector<std::vector<Point>> population(populationSize);

        const size_t verticesSize = vertices.size();
        const size_t verticesSizeMinusOne = verticesSize - 1;

#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(populationSize); ++i)
        {
            population[i] = vertices;

            auto nearestPointIt = std::min_element(population[i].begin(), population[i].end(),
                [fixedPoint](const Point& a, const Point& b)
                {
                    return CalculateDistance(fixedPoint, a) < CalculateDistance(fixedPoint, b);
                });

            std::iter_swap(population[i].begin(), nearestPointIt);
            std::shuffle(population[i].begin() + 1, population[i].end() - 1, rng);
        }

        return population;
    }

    /// <summary>
    /// Evolves the given population through a specified number of generations.
    /// </summary>
    /// <param name="population">The initial population of routes.</param>
    /// <param name="generations">The number of generations to evolve.</param>
    /// <param name="tournamentSize">The size of the tournament for parent selection.</param>
    /// <param name="initialMutationRate">The initial mutation rate.</param>
    static void EvolvePopulation(std::vector<std::vector<Point>>& population, size_t generations, size_t tournamentSize, float initialMutationRate)
    {
        float mutationRate = initialMutationRate;

        std::random_device rd;
        std::mt19937 g(rd());

        auto calculateTotalDistances = [&](const std::vector<std::vector<Point>>& pop)
            {
                std::vector<float> distances(pop.size());
                const size_t popSize = pop.size();

#pragma omp parallel for
                for (int i = 0; i < static_cast<int>(popSize); ++i)
                {
                    distances[i] = CalculateTotalPathDistance(pop[i]);
                }

                return distances;
            };

        std::vector<float> distances = calculateTotalDistances(population);
        float bestDistance = *std::min_element(distances.begin(), distances.end());

        for (size_t generation = 0; generation < generations; ++generation)
        {
            std::vector<std::vector<Point>> newPopulation(population.size());

#pragma omp parallel for
            for (int i = 0; i < static_cast<int>(population.size()); ++i)
            {
                if (i == 0)
                {
                    // If it's the first std::vector<Point> (elite), enforce the fixed starting point
                    newPopulation[i] = population[std::min_element(distances.begin(), distances.end()) - distances.begin()];
                    continue;
                }

                auto [parent1, parent2] = TournamentSelection(population, tournamentSize, g);

                std::vector<Point> child = Crossover(parent1, parent2, g);
                Mutate(child, mutationRate, g);

                newPopulation[i] = std::move(child);
            }

#pragma omp parallel for
            for (int i = 1; i < static_cast<int>(population.size()); ++i)
            {
                LocalSearch(newPopulation[i]);
                Mutate(newPopulation[i], mutationRate, g);
            }

            population = std::move(newPopulation);

            distances = calculateTotalDistances(population);
            bestDistance = *std::min_element(distances.begin(), distances.end());

            mutationRate = initialMutationRate * (1.0f - static_cast<float>(generation) / generations);
        }
    }

    /// <summary>
    /// Finds the best route using the genetic algorithm after evolving the population.
    /// </summary>
    /// <param name="points">The set of points representing the vertices of the problem.</param>
    /// <param name="fixedPoint">The fixed starting point for all routes.</param>
    /// <param name="populationSize">The size of the population used in the genetic algorithm. Default is 25.</param>
    /// <param name="generations">The number of generations for which the genetic algorithm evolves the population. Default is 100.</param>
    /// <param name="tournamentSize">The size of the tournament used in the genetic algorithm for parent selection. Default is 5.</param>
    /// <param name="mutationRate">The mutation rate used in the genetic algorithm. Default is 0.02.</param>
    /// <returns>The best route found by the genetic algorithm.</returns>
    static std::vector<Point> FindPath(const std::vector<Point>& points, const Point& fixedPoint, size_t populationSize = 25, size_t generations = 100, size_t tournamentSize = 5, float mutationRate = 0.02) noexcept
    {
        std::mt19937 rng(std::random_device{}());
        std::vector<std::vector<Point>> population = InitializePopulation(points, populationSize, rng, fixedPoint);

        EvolvePopulation(population, generations, tournamentSize, mutationRate);

        return *std::min_element(population.begin(), population.end(),
            [](const std::vector<Point>& a, const std::vector<Point>& b)
            {
                return CalculateTotalPathDistance(a) < CalculateTotalPathDistance(b);
            });
    }
}
