#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <string>
#include <limits>
#include <ctime>
#include <cmath>

const int _OUTPUT_FILE_OPEN_ERROR_ = -100;
const int _INPUT_FILE_OPEN_ERROR_ = -101;
const int _UNKNOWN_TOPOLOGY_TYPE_ = -102;
const int _BADMOVE_EXCEPTION_ = -103;
const int _GT_ALLOWABLE_LIMIT_ = -104;
const int _DICTIONARY_FILE_OPEN_ERROR = -105;

const int _HELP_CALLED_ = 101;

typedef unsigned int		uint;
typedef unsigned char		uchar;

struct Point {
	int x;
	int y;
};

struct Config {
	bool help = false;
	std::string space = "space.txt";
	std::string out = "route.txt";
	std::string dict = "dictionary.txt";
	int limit = 1000;
	std::string topology = "planar";
};

class BadMove : public std::exception {};
class FileError : public std::exception {};

Config ReadParams(int argc, char** argv);
void CallHelp();

template <typename P, typename M>
class Surface
{
public:
	virtual M Distance(P &from, P &to) = 0;
	virtual std::tuple<P, M> GetClosest(std::vector<std::tuple<P, M>> &lookupResults) = 0;

	virtual M Move(P &point) = 0;
	virtual std::vector<std::tuple<P, M>> LookUp() = 0;
	virtual void show(std::ostream &to) = 0;
};

class Planar : public Surface <Point, uint>
{
public:
	Point start, finish, curr;
	std::vector<std::vector<uchar>> matrix;
	Planar(std::istream& from);
	~Planar();

	virtual uint Distance(Point &from, Point &to);
	std::tuple<Point, uint> GetClosest(std::vector<std::tuple<Point, uint>> &lookupResults);

	virtual uint Move(Point &point);
	virtual std::vector<std::tuple<Point, uint>> LookUp();

	virtual void show(std::ostream &to);

	/*
	virtual std::ofstream& operator<<(std::ofstream &to);
	virtual std::ifstream& operator >> (std::ifstream &from);*/
};

class Dictionary : public Surface <std::string, uint> {
public:
	std::string start, finish, curr, dict;
	std::vector<std::vector<std::string>> cache;
	std::vector<std::string> path;
	bool isRead;

	Dictionary(std::istream &from, std::string dictFile);
	~Dictionary();

	uint Distance(std::string &from, std::string &to);
	std::tuple<std::string, uint> GetClosest(std::vector<std::tuple<std::string, uint>> &lookupResults);

	uint Move(std::string &word) ;
	std::vector<std::tuple<std::string, uint>> LookUp();

	virtual void show(std::ostream &to);

	/*
	virtual std::ofstream& operator<<(std::ofstream &to);
	virtual std::ifstream& operator >> (std::ifstream &from);*/
};

template<class P, class M>
class Robot {
	Surface<P, M> *space;

public:
	Robot(Surface<P, M> *space) : space(space) {};
	Surface<P,M>* getSapce() {
		return this->space;
	}
	void Discover() {
		std::tuple<P, M> next;
		P pNext;
		M dist;

		do
		{
			next = space->GetClosest(space->LookUp());
			pNext = std::get<0>(next);

			dist = space->Move(pNext);

		} while (dist);
	};
	//void ShowPath(std::ostream &to) {
		//this->space->show(to);
		//to << *(this->space);
	//};

};

std::ostream& operator<<(std::ostream &to, Planar &pl);
std::istream& operator >> (std::istream &from, Planar &pl);

std::ostream& operator<<(std::ostream &to, Dictionary &space);
std::istream& operator >> (std::istream &from, Dictionary &space);
