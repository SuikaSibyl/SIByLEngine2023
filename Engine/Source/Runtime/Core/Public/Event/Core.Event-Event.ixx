export module Core.Event:Event;

namespace SIByL::Core
{
	export enum struct EventType
	{
		None,
		WindowClose, WindowResize, WindowFocus, WindowLostFocus, WindowMoved,
		KeyPressed, KeyReleased, KeyTyped,
		MouseButtonPressed, MouseButtonReleased, MouseMoved, MouseScrolled,
	};

	export enum struct EventCategory
	{

	};
}