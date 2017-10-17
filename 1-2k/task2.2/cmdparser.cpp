#include "Surface.h"

Config ReadParams(int argc, char** argv)
{
	Config result;

	if (argc == 1)
		result.help = true;

	for (uint i = 1; i < (uint)argc; i++) {
		if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
		{
			result.help = true;
			break;
		}

		if (!strcmp(argv[i], "-s") || !strcmp(argv[i], "--space"))
		{
			result.space = argv[++i];
			continue;
		}

		if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--out"))
		{
			result.out = argv[++i];
			continue;
		}

		if (!strcmp(argv[i], "-l") || !strcmp(argv[i], "--limit"))
		{
			result.limit = atoi(argv[++i]);
			continue;
		}

		if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "--topology"))
		{
			result.topology = argv[++i];
			continue;
		}
		if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--dictionary"))
		{
			result.dict = argv[++i];
			continue;
		}
	}

	return result;
}

void CallHelp() {
	std::cout << "(-h --help) - вызвать хелп" << std::endl;
	std::cout << "(-s --space) - путь к файлу спейс" << std::endl;
	std::cout << "(-o --out) - путь к файлу вывода" << std::endl;
	std::cout << "(-l --limit) - значение лимита" << std::endl;
	std::cout << "(-d --dictionary) - путь к файлу со словарем" << std::endl;
	std::cout << "(-t --topology) - топология" << std::endl;
	std::cout << "\tplanar - планарное поле" << std::endl;
	std::cout << "\tdict - словарь" << std::endl;
}