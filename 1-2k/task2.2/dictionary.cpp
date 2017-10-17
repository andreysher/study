#include "Surface.h"

Dictionary::Dictionary(std::istream &from, std::string dictFile)
{
	from >> *this;

	this->dict = dictFile;
	this->isRead = false;
	this->cache.reserve(Distance(start, finish) - 1);

	this->path.push_back(start);
}

void Dictionary::show(std::ostream& to) {
	to << *this;
}

Dictionary::~Dictionary()
{
	cache.clear();
	path.clear();

	cache.~vector();
	path.~vector();
}

size_t Dictionary::Distance(std::string &from, std::string &to)
{
	const uint len1 = from.size(),
		len2 = to.size();

	std::vector<uint> col(len2 + 1),
		prevCol(len2 + 1);

	for (uint i = 0; i < prevCol.size(); i++)
	{
		prevCol[i] = i;
	}

	for (uint i = 0; i < len1; i++)
	{
		col[0] = i + 1;

		for (uint j = 0; j < len2; j++)
		{
			col[j + 1] = (uint)fmin(fmin(prevCol[1 + j] + 1, col[j] + 1), prevCol[j] + (from[i] == to[j] ? 0 : 1));
		}
		col.swap(prevCol);
	}

	return prevCol[len2];
}

std::vector<std::tuple<std::string, uint>> Dictionary::LookUp()
{
	std::vector<std::tuple<std::string, uint>> result;

	size_t maxDest = Distance(start, finish);
	size_t distance = SIZE_MAX;
	std::ifstream dictFile;
	std::string word;
	uint currInd;

	if (!isRead)
	{
		dictFile.open(dict);

		if (!dictFile.good())
		{
			throw FileError();
		}
		while (!dictFile.eof())
		{
			dictFile >> word;

			if (((distance = Distance(start, word)) > 0) && (distance <= cache.size()) && (Distance(word, finish) < Distance(start, finish)))
			{
				cache[distance - 1].push_back(word);
			}

		}

		dictFile.close();
		isRead = true;
	}

	distance = Distance(curr, finish);
	currInd = Distance(start, finish) - distance;

	for (uint i = 0; i < cache[currInd].size(); ++i)
	{
		result.push_back(make_tuple(cache[currInd][i], currInd + 1));
	}

	return result;
}

std::tuple<std::string, uint> Dictionary::GetClosest(std::vector<std::tuple<std::string, uint>> &lookupResults)
{
	uint dist, minDist = SIZE_MAX;
	std::string needStr;

	if (!lookupResults.size())
	{
		return (std::make_tuple(start, 0));
	}
	else {
		for (uint i = 0; i < lookupResults.size(); ++i)
		{
			dist = Distance(std::get<0>(lookupResults[i]), finish);

			if (dist <= minDist)
			{
				minDist = dist;
				needStr = std::get<0>(lookupResults[i]);

				if (std::get<0>(lookupResults[i]) == finish)
				{
					break;
				}
			}
		}
	}

	return (make_tuple(needStr, minDist));
}

uint Dictionary::Move(std::string &word) {
	if (word == start || word == curr)
	{
		throw BadMove();
	}

	curr = word;
	path.push_back(word);

	return Distance(word, finish);
}

std::ostream& operator<<(std::ostream& to, Dictionary &space)
{
	for (uint i = 0; i < space.path.size(); ++i)
	{
		to << " " << space.path[i] << ((i == space.path.size() - 1) ? "" : " ->");
	}

	return to;
}

std::istream& operator >> (std::istream &from, Dictionary &space)
{
	from >> space.start;
	from >> space.finish;
	space.curr = space.start;

	space.cache = std::vector<std::vector<std::string>>(space.Distance(space.start, space.finish));

	return from;
}