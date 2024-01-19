#pragma once

struct Point
{
    float x, y;

    Point() : x(0.0f), y(0.0f) {}
    Point(float x, float y) : x(x), y(y) {}

    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};
