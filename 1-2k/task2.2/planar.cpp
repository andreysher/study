#include "Surface.h"

Planar::Planar(std::istream &from)
{
	from >> *this;
}

void Planar::show(std::ostream& to) {
	to << *this;
}

Planar::~Planar()
{
	matrix.clear();
	matrix.~vector();
}

uint Planar::Distance(Point &from, Point &to)
{
	return abs(from.x - to.x) + abs(from.y - to.y);
}


std::vector<std::tuple<Point, uint>> Planar::LookUp()
{
	std::vector<std::tuple<Point, uint>> res;
	uint height = matrix.size();

	Point points[] =
	{
		{ curr.x - 1, curr.y },		// Left Point
		{ curr.x, curr.y - 1 },		// Up Point
		{ curr.x + 1, curr.y },		// Right Point
		{ curr.x, curr.y + 1 }		// Bottom Point
	};

	for (Point p : points)
	{
		if (p.x >= 0 && p.y >= 0 && p.y < (int)height && p.x < (int)matrix[p.y].size())
		{
			if (matrix[p.y][p.x] == '.' || matrix[p.y][p.x] == 'F')
			{
				res.push_back(std::make_tuple(p, Distance(p, finish)));
			}
		}
	}

	return res;
}

std::tuple<Point, uint> Planar::GetClosest(std::vector<std::tuple<Point, uint>> &lookupRes)
{
	size_t min = SIZE_MAX;
	size_t dest = 0;
	char needInd = 0;

	if (!lookupRes.capacity())
	{
		return std::make_tuple(start, SIZE_MAX);
	}

	for (char i = 0; i < (char)lookupRes.capacity(); i++)
	{
		dest = std::get<1>(lookupRes[i]);

		if (dest < min)
		{
			min = dest;
			needInd = i;
		}
	}
	return lookupRes[needInd];
}
uint Planar::Move(Point &p)
{
	if (p.x == start.x && p.y == start.y)
	{
		throw BadMove();
	}

	curr = p;

	if (curr.x != finish.x || curr.y != finish.y)
		matrix[curr.y][curr.x] = '*';

	return Distance(curr, finish);
}

std::ostream& operator<<(std::ostream& to, Planar &space)
{
	for (size_t row = 0; row < space.matrix.size(); row++)
	{
		for (size_t col = 0; col < space.matrix[row].size(); col++)
		{
			to << space.matrix[row][col] << " ";
		}
		to << std::endl;
	}

	return to;
}

std::istream& operator >> (std::istream &from, Planar &space)
{
	std::string line;
	uint col, row;
	std::vector<uchar> tmpVec;

	col = row = 0;

	while (!from.eof())
	{
		getline(from, line);

		for (uchar ch : line)
		{
			switch (ch)
			{
			case 'S':
				space.start.x = space.curr.x = col;
				space.start.y = space.curr.y = row;
				break;

			case 'F':
				space.finish.x = col;
				space.finish.y = row;
				break;

			default:
				break;
			}

			tmpVec.push_back(ch);
			col += 1;
		}

		space.matrix.push_back(tmpVec);
		row += 1;
		col = 0;
		tmpVec.clear();
	}

	return from;
}

