#include "Main.hpp"

int main(int argc, char* argv[]) {
    constexpr float fps = 60.0f;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("SDL2 - Playground", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 600, 600, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Event e{ 0 };
    Point mousePos{ 0, 0 };

    std::vector<Point> vertices =
    {
        {30.0f, 10.0f},
        {130.0f, 50.0f},
        {200.0f, 30.0f},
        {550.0f, 160.0f},
        {400.0f, 400.0f},
        {500.0f, 500.0f},
        {500.0f, 560.0f},
        {120.0f, 500.0f},
        {300.0f, 200.0f},
    };

    bool lclick = false;
    bool rclick = false;
    bool mclick = false;
    Polygon concavePolygon(vertices);

    while (true)
    {
        if (SDL_PollEvent(&e))
        {
            if (e.type == SDL_KEYDOWN || e.type == SDL_QUIT)
            {
                break;
            }
            else if (e.type == SDL_MOUSEBUTTONUP)
            {
                if (e.button.button == SDL_BUTTON_LEFT)
                {
                    lclick = true;
                }
                else if (e.button.button == SDL_BUTTON_RIGHT)
                {
                    rclick = true;
                }
                else if (e.button.button == SDL_BUTTON_MIDDLE)
                {
                    mclick = true;
                }
            }
            else if (e.type == SDL_MOUSEMOTION)
            {
                mousePos.x = static_cast<float>(e.motion.x);
                mousePos.y = static_cast<float>(e.motion.y);
            }
        }

        Render(window, renderer, mousePos, concavePolygon, lclick, rclick, mclick);
        lclick = false;
        rclick = false;
        mclick = false;

        SDL_Delay(static_cast<Uint32>(1000.0f / fps));
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

static void Render(SDL_Window* window, SDL_Renderer* renderer, const Point& mousePos, const Polygon& poly, bool lclick, bool rclick, bool mclick) noexcept
{
    float viewDistance = 25.0f;
    static std::vector<Point> points;
    static std::vector<std::tuple<std::vector<Point>, std::tuple<int, int, int>, std::string>> paths;
    static int renderIndex = 0;

    SDL_SetRenderDrawColor(renderer, 46, 46, 46, 255);
    SDL_RenderClear(renderer);

    // Render polygon
    SDL_SetRenderDrawColor(renderer, 214, 214, 214, 255);

    for (size_t i = 0; i < poly.vertices.size(); ++i)
    {
        const Point& p1 = poly.vertices[i];
        const Point& p2 = poly.vertices[(i + 1) % poly.vertices.size()];
        SDL_RenderDrawLine(renderer, p1.x, p1.y, p2.x, p2.y);
    }

    // Sample points on polygon
    if (mclick || points.empty())
    {
        points = poly.BridsonsPoissonDiskSampling(viewDistance);
        paths.clear();
    }

    SDL_SetRenderDrawColor(renderer, 255, 165, 0, 255);

    for (const auto& p : points)
    {
        SDL_Rect dotRect{ static_cast<int>(p.x) - 2, static_cast<int>(p.y) - 2, 4, 4 };
        SDL_RenderFillRect(renderer, &dotRect);
    }

    // Generate path using TSP solver
    if (lclick || paths.empty())
    {
        paths.clear();

        int algCount = 6;

        // sorted by best results

        // ILS: very compute intensive, very good results
        std::vector<Point> pathIls;
        const auto ilsTime = measureExecutionTime([&pathIls, mousePos]() { pathIls = IteratedLocalSearch::FindPath(points, mousePos); });
        paths.push_back(std::make_tuple(pathIls, std::make_tuple(158, 134, 200), std::format("(1/{0}) Iterated Local Search: {1}ms, {2}m", algCount, ilsTime, CalculateTotalPathDistance(pathIls))));

        // GA: compute intensive, vergy good results
        std::vector<Point> pathGa;
        const auto gaTime = measureExecutionTime([&pathGa, mousePos]() { pathGa = GeneticAlgorithm::FindPath(points, mousePos); });
        paths.push_back(std::make_tuple(pathGa, std::make_tuple(180, 210, 115), std::format("(2/{0}) Genetic Algorithm: {1}ms, {2}m", algCount, gaTime, CalculateTotalPathDistance(pathGa))));

        // ACO: fast, good results
        std::vector<Point> pathAco;
        const auto acoTime = measureExecutionTime([&pathAco, mousePos]() { pathAco = AntColonyOptimization::FindPath(points, mousePos); });
        paths.push_back(std::make_tuple(pathAco, std::make_tuple(232, 125, 62), std::format("(3/{0}) Ant Colony Optimization: {1}ms, {2}m", algCount, acoTime, CalculateTotalPathDistance(pathAco))));

        // NN: very fast, good results
        std::vector<Point> pathNn;
        const auto nnTime = measureExecutionTime([&pathNn, mousePos]() { pathNn = NearestNeighbor::FindPath(points, mousePos); });
        paths.push_back(std::make_tuple(pathNn, std::make_tuple(108, 153, 187), std::format("(4/{0}) Nearest Neighbor: {1}ms, {2}m", algCount, nnTime, CalculateTotalPathDistance(pathNn))));

        // SA: very fast, unuseable results
        std::vector<Point> pathSa;
        const auto saTime = measureExecutionTime([&pathSa, mousePos]() { pathSa = SimulatedAnnealing::FindPath(points, mousePos); });
        paths.push_back(std::make_tuple(pathSa, std::make_tuple(176, 82, 121), std::format("(5/{0}) Simulated Annealing: {1}ms, {2}m", algCount, saTime, CalculateTotalPathDistance(pathSa))));
        
        // PSO: very fast, unuseable results
        std::vector<Point> pathPso;
        const auto psoTime = measureExecutionTime([&pathPso, mousePos]() { pathPso = ParticleSwarmOptimization::FindPath(points, mousePos); });
        paths.push_back(std::make_tuple(pathPso, std::make_tuple(229, 181, 103), std::format("(6/{0}) Particle Swarm Optimization: {1}ms, {2}m", algCount, psoTime, CalculateTotalPathDistance(pathPso))));

        // HK: disabled as its not working atm
        // std::vector<Point> pathHk;
        // const auto hkTime = measureExecutionTime([&pathHk, mousePos]() { pathHk = HeldKarp::FindPath(points, mousePos); });
        // paths.push_back(std::make_tuple(pathHk, std::make_tuple(229, 181, 103), std::format("(7/{0}) Held-Karp: {1}ms, {2}m", algCount, hkTime, CalculateTotalPathDistance(pathHk))));
    }

    // Render path
    if (rclick)
    {
        renderIndex++;

        if (renderIndex > paths.size() - 1)
        {
            renderIndex = 0;
        }
    }

    if (!paths.empty())
    {
        const auto& path = std::get<0>(paths[renderIndex]);
        const auto nodeCount = path.size() > 1 ? path.size() - 1 : 0;

        if (nodeCount >= 0)
        {
            const auto& color = std::get<1>(paths[renderIndex]);
            const auto& algorithmName = std::get<2>(paths[renderIndex]);

            SDL_SetRenderDrawColor(renderer, std::get<0>(color), std::get<1>(color), std::get<2>(color), 255);
            SDL_SetWindowTitle(window, algorithmName.c_str());

            for (int i = 0; i < nodeCount; ++i)
            {
                SDL_RenderDrawLine(renderer, path[i].x, path[i].y, path[i + 1].x, path[i + 1].y);
            }
        }
    }

    // Mouse point check in polygon
    if (poly.IsInside(mousePos))
    {
        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
    }
    else
    {
        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    }

    SDL_Rect mouseRect{ static_cast<int>(mousePos.x) - 2, static_cast<int>(mousePos.y) - 2, 4, 4 };
    SDL_RenderFillRect(renderer, &mouseRect);

    SDL_RenderPresent(renderer);
}
