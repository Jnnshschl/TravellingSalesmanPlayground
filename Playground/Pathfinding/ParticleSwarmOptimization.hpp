#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "../Misc/Point.hpp"
#include "../Misc/MathHelpers.hpp"

namespace ParticleSwarmOptimization 
{
    struct Particle
    {
        std::vector<Point> route;
        float fitness;
        std::vector<Point> personalBest;
        float personalBestFitness;
        std::vector<Point> velocity;
    };

    /// <summary>
    /// Calculates the total distance of a given route.
    /// </summary>
    /// <param name="route">The route to calculate the distance for.</param>
    /// <returns>The total distance of the route.</returns>
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

    /// <summary>
    /// Initializes a particle with a route and other properties.
    /// </summary>
    /// <param name="particle">The particle to initialize.</param>
    /// <param name="initialRoute">The initial route for the particle.</param>
    /// <param name="start">The starting point for the route.</param>
    static void InitializeParticle(Particle& particle, const std::vector<Point>& initialRoute, const Point& start)
    {
        particle.route = initialRoute;
        auto nearestStart = FindNearestPointInList(initialRoute, start);
        auto it = std::find(particle.route.begin(), particle.route.end(), nearestStart);
        std::rotate(particle.route.begin(), it, particle.route.end());

        particle.fitness = CalculateRouteCost(particle.route);
        particle.personalBest = particle.route;
        particle.personalBestFitness = particle.fitness;

        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        particle.velocity.resize(initialRoute.size());
        for (size_t i = 0; i < initialRoute.size(); ++i)
        {
            particle.velocity[i].x = dist(rng);
            particle.velocity[i].y = dist(rng);
        }
    }

    /// <summary>
    /// Updates the particle's position, velocity, and fitness based on PSO equations.
    /// </summary>
    /// <param name="particle">The particle to update.</param>
    /// <param name="globalBest">The global best particle.</param>
    /// <param name="inertiaWeight">The inertia weight parameter.</param>
    /// <param name="personalWeight">The personal weight parameter.</param>
    /// <param name="globalWeight">The global weight parameter.</param>
    /// <returns>The updated particle.</returns>
    static Particle UpdateParticle(const Particle& particle, const Particle& globalBest, float inertiaWeight, float personalWeight, float globalWeight)
    {
        Particle updatedParticle = particle;
        updatedParticle.velocity.resize(particle.route.size());

        for (size_t i = 0; i < particle.velocity.size(); ++i)
        {
            float inertiaTerm = inertiaWeight * particle.velocity[i].x;
            float personalBestTerm = personalWeight * (particle.personalBest[i].x - particle.route[i].x);
            float globalBestTerm = globalWeight * (globalBest.route[i].x - particle.route[i].x);

            updatedParticle.velocity[i].x = inertiaTerm + personalBestTerm + globalBestTerm;

            inertiaTerm = inertiaWeight * particle.velocity[i].y;
            personalBestTerm = personalWeight * (particle.personalBest[i].y - particle.route[i].y);
            globalBestTerm = globalWeight * (globalBest.route[i].y - particle.route[i].y);

            updatedParticle.velocity[i].y = inertiaTerm + personalBestTerm + globalBestTerm;
        }

        for (size_t i = 0; i < particle.route.size(); ++i)
        {
            updatedParticle.route[i].x += updatedParticle.velocity[i].x;
            updatedParticle.route[i].y += updatedParticle.velocity[i].y;
        }

        updatedParticle.fitness = CalculateRouteCost(updatedParticle.route);

        if (updatedParticle.fitness < updatedParticle.personalBestFitness)
        {
            updatedParticle.personalBest = updatedParticle.route;
            updatedParticle.personalBestFitness = updatedParticle.fitness;
        }

        return updatedParticle;
    }

    /// <summary>
    /// Finds the global best particle in the swarm.
    /// </summary>
    /// <param name="swarm">The swarm of particles.</param>
    /// <returns>The global best particle.</returns>
    static Particle FindGlobalBest(const std::vector<Particle>& swarm)
    {
        return *std::min_element(swarm.begin(), swarm.end(), [](const Particle& p1, const Particle& p2)
        {
            return p1.personalBestFitness < p2.personalBestFitness;
        });
    }

    /// <summary>
    /// Finds the optimized path using Particle Swarm Optimization (PSO).
    /// </summary>
    /// <param name="initialRoute">The initial route of points.</param>
    /// <param name="start">The starting point for the route.</param>
    /// <param name="inertiaWeight">The inertia weight parameter.</param>
    /// <param name="personalWeight">The personal weight parameter.</param>
    /// <param name="globalWeight">The global weight parameter.</param>
    /// <param name="swarmSize">The size of the particle swarm.</param>
    /// <param name="maxIterations">The maximum number of iterations.</param>
    /// <returns>The optimized route.</returns>
    static std::vector<Point> FindPath(const std::vector<Point>& initialRoute, const Point& start, float inertiaWeight = 0.5f, float personalWeight = 1.5f, float globalWeight = 1.5f, int swarmSize = 50, int maxIterations = 100)
    {
        std::vector<Particle> swarm(swarmSize);

        for (int i = 0; i < swarmSize; ++i)
        {
            InitializeParticle(swarm[i], initialRoute, start);
        }

        Particle globalBest = FindGlobalBest(swarm);

        for (int iteration = 0; iteration < maxIterations; ++iteration)
        {
            for (auto& particle : swarm)
            {
                particle = UpdateParticle(particle, globalBest, inertiaWeight, personalWeight, globalWeight);
            }

            globalBest = FindGlobalBest(swarm);
        }

        return globalBest.route;
    }
}
