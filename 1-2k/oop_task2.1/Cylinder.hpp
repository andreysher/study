#pragma once
#include "header.hpp"
#include "Space.hpp"

class Cylinder : public Space {
public:
	Cylinder(istream &from);
	Cylinder() {};
	~Cylinder();

	size_t dest(Point &from, Point &to);
	vector<tuple<Point, size_t>> lookup();
};