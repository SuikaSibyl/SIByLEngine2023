export module Application.Root;
import Core.Log;
import Core.Memory;

namespace SIByL::Application
{
	export struct Root
	{
		Root();
		~Root();

		Core::MemoryManager gMemManager;
		Core::LogManager	gLogManager;
	};

	Root::Root()
	{
		gMemManager.startUp();
		gLogManager.startUp();
	}

	Root::~Root()
	{
		gLogManager.shutDown();
		gMemManager.shutDown();
	}

}