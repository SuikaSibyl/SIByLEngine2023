module;
#include <cmath>
export module Tracer.Spectrum:CoefficientSpectrum;
import Math.Limits;
import Math.Common;

namespace SIByL::Tracer
{
	/**
	* Represent a spectrum as a particular number of samples
	* Impose the implicit assumption that the spectral representation is a set of coefficients 
	* that linearly scale a fixed set of basis function.
	*/
	export template<size_t nSpectrumSamples>
	struct CoefficientSpectrum
	{
	public:
		CoefficientSpectrum(float v = 0.f);

		auto operator-()->CoefficientSpectrum&;
		auto operator+=(CoefficientSpectrum const& s2)->CoefficientSpectrum&;
		auto operator-=(CoefficientSpectrum const& s2)->CoefficientSpectrum&;
		auto operator*=(CoefficientSpectrum const& s2)->CoefficientSpectrum&;
		auto operator/=(CoefficientSpectrum const& s2)->CoefficientSpectrum&;
		auto operator+(CoefficientSpectrum const& s2) const->CoefficientSpectrum;
		auto operator-(CoefficientSpectrum const& s2) const->CoefficientSpectrum;
		auto operator*(CoefficientSpectrum const& s2) const->CoefficientSpectrum;
		auto operator/(CoefficientSpectrum const& s2) const->CoefficientSpectrum;
		auto operator+(float x) const->CoefficientSpectrum;
		auto operator-(float x) const->CoefficientSpectrum;
		auto operator*(float x) const->CoefficientSpectrum;
		auto operator/(float x) const->CoefficientSpectrum;

		auto isBlack() const noexcept -> bool;
		auto hasNaNs() const noexcept -> bool;
		auto clamp(float low = 0, float high = Math::float_max) noexcept -> CoefficientSpectrum;

		friend inline auto sqrt(CoefficientSpectrum const& s) noexcept -> CoefficientSpectrum;
		friend inline auto lerp(float t, CoefficientSpectrum const& s1, CoefficientSpectrum const& s2) noexcept -> CoefficientSpectrum;
		friend inline auto clamp(CoefficientSpectrum const& s, float low = 0, float high = Math::float_max) noexcept -> CoefficientSpectrum;
		friend auto operator*(float v, CoefficientSpectrum const& s)->CoefficientSpectrum;

		static size_t const nSamples = nSpectrumSamples;
		auto operator[](size_t i) -> float& { return c[i]; }

	protected:
		float c[nSpectrumSamples];
	};

	/**
	* Template Member Func Impl
	*/
	template<size_t nSpectrumSamples>
	CoefficientSpectrum<nSpectrumSamples>::CoefficientSpectrum(float v)
	{
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			c[i] = v;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator-()->CoefficientSpectrum&
	{
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			c[i] = -c[i];
		return *this;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator+=(CoefficientSpectrum const& s2)->CoefficientSpectrum&
	{
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			c[i] += s2.c[i];
		return *this;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator-=(CoefficientSpectrum const& s2)->CoefficientSpectrum&
	{
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			c[i] -= s2.c[i];
		return *this;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator*=(CoefficientSpectrum const& s2)->CoefficientSpectrum&
	{
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			c[i] *= s2.c[i];
		return *this;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator/=(CoefficientSpectrum const& s2)->CoefficientSpectrum&
	{
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			c[i] /= s2.c[i];
		return *this;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator+(CoefficientSpectrum const& s2) const->CoefficientSpectrum
	{
		CoefficientSpectrum<nSpectrumSamples> ret = *this;
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			ret.c[i] += s2.c[i];
		return ret;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator-(CoefficientSpectrum const& s2) const->CoefficientSpectrum
	{
		CoefficientSpectrum<nSpectrumSamples> ret = *this;
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			ret.c[i] -= s2.c[i];
		return ret;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator*(CoefficientSpectrum const& s2) const->CoefficientSpectrum
	{
		CoefficientSpectrum<nSpectrumSamples> ret = *this;
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			ret.c[i] *= s2.c[i];
		return ret;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator/(CoefficientSpectrum const& s2) const->CoefficientSpectrum
	{
		CoefficientSpectrum<nSpectrumSamples> ret = *this;
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			ret.c[i] /= s2.c[i];
		return ret;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator+(float x) const->CoefficientSpectrum<nSpectrumSamples>
	{
		CoefficientSpectrum<nSpectrumSamples> ret = *this;
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			ret.c[i] += x;
		return ret;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator-(float x) const->CoefficientSpectrum<nSpectrumSamples>
	{
		CoefficientSpectrum<nSpectrumSamples> ret = *this;
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			ret.c[i] -= x;
		return ret;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator*(float x) const->CoefficientSpectrum<nSpectrumSamples>
	{
		CoefficientSpectrum<nSpectrumSamples> ret = *this;
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			ret.c[i] *= x;
		return ret;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::operator/(float x) const->CoefficientSpectrum<nSpectrumSamples>
	{
		CoefficientSpectrum<nSpectrumSamples> ret = *this;
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			ret.c[i] /= x;
		return ret;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::isBlack() const noexcept -> bool
	{
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			if (c[i] != 0.) return false;
		return true;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::hasNaNs() const noexcept -> bool
	{
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			if (std::isnan(c[i])) return true;
		return false;
	}

	template<size_t nSpectrumSamples>
	auto CoefficientSpectrum<nSpectrumSamples>::clamp(float low, float high) noexcept -> CoefficientSpectrum
	{
		CoefficientSpectrum<nSpectrumSamples> ret;
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			ret.c[i] = Math::clamp(c[i], low, high);
		return ret;
	}

	/**
	* Template Global Func Impl
	*/
	export template<size_t nSpectrumSamples>
	auto operator*(float v, CoefficientSpectrum<nSpectrumSamples> const& s) -> CoefficientSpectrum<nSpectrumSamples>
	{
		CoefficientSpectrum<nSpectrumSamples> ret;
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			ret[i] = v * s.c[i];
		return ret;
	}

	export template<size_t nSpectrumSamples>
		inline auto sqrt(CoefficientSpectrum<nSpectrumSamples> const& s) noexcept -> CoefficientSpectrum<nSpectrumSamples>
	{
		CoefficientSpectrum<nSpectrumSamples> ret;
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			ret.c[i] = std::sqrt(s.c[i]);
		return ret;
	}

	export template<size_t nSpectrumSamples>
		inline auto lerp(float t, CoefficientSpectrum<nSpectrumSamples> const& s1, CoefficientSpectrum<nSpectrumSamples> const& s2) noexcept -> CoefficientSpectrum<nSpectrumSamples>
	{
		return s1 * (1 - t) + s2 * t;
	}

	export template<size_t nSpectrumSamples>
		inline auto clamp(CoefficientSpectrum<nSpectrumSamples> const& s, float low = 0, float high = Math::float_infinity) noexcept -> CoefficientSpectrum<nSpectrumSamples>
	{
		CoefficientSpectrum<nSpectrumSamples> ret;
		for (size_t i = 0; i < nSpectrumSamples; ++i)
			ret.c[i] = Math::clamp(s.c[i], low, high);
		return ret;
	}
}