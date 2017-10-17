#include "Surface.h"

int main(int argc, char** argv)
{

	setlocale(0, "Russian");

	Config params;
	uint dist = 0;
	uint ind = 0;
	std::ifstream from;
	std::ofstream to;

	params = ReadParams(argc, argv);

	if (params.help)
	{
		CallHelp();
		return _HELP_CALLED_;
	}

	from.open(params.space);
	if (!from.good())
	{
		std::cout << "не открылся спейс" << "\"" << params.space << "\"" << std::endl;
		return _INPUT_FILE_OPEN_ERROR_;
	}

	to.open(params.out);
	if (!to.good())
	{
		std::cout << "не открылся аут" << "\"" << params.out << "\"" << std::endl;
		return _OUTPUT_FILE_OPEN_ERROR_;
	}

	try
	{
		if (params.topology == "planar")
		{
			Planar space(from);
			Robot<Point, uint> rbt(&space);
			rbt.Discover();
			(rbt.getSapce())->show(to);
		}
		else if (params.topology == "dict")
		{
			Dictionary space(from,params.dict);
			from >> space;
			Robot<std::string, uint> rbt(&space);
			rbt.Discover();
			(rbt.getSapce())->show(to);
			//rbt.ShowPath(to);
		}
	}
	catch (FileError)
	{
		std::cout << "не открылся словарь" << "\"" << params.dict << "\"" << std::endl;
		return _DICTIONARY_FILE_OPEN_ERROR;
	}
	catch (BadMove)
	{
		std::cout << "нет пути" << std::endl;
		return _BADMOVE_EXCEPTION_;
	}


	return 0;
}