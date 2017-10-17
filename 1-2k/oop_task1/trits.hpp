#ifndef _TRITS_LIB
#define _TRITS_LIB

#include <unordered_map>

typedef unsigned int uint;
typedef unsigned short ushort;

const ushort _TRITS_IN_UINT_ = 4 * sizeof(uint);	

const uint _FALSE_ = 0x0;
const uint _UNKNOWN_ = 0x1;
const uint _TRUE_ = 0x3;

const uint _UNKNOWN_BLOCK_ = 0x55555555;

namespace Trits
{
	enum Trit { False, Unknown, True };

	class TritSet
	{
		size_t numOfTrits;		
		size_t arrLen;			
		uint *trits = nullptr;
        class TritHelp
	    {
			TritSet *parent;		
			size_t arrIndex;		
			ushort tritIndex;		
			Trit trit;				

		public:

			TritHelp(TritSet *parent, size_t index);
			~TritHelp();
			

			Trit operator=(Trit val);
			Trit operator=(TritHelp& rp);
			operator Trit();


			Trit getTrit(size_t index);
			void setTrit(Trit val);
		};


	public:
		TritSet(size_t numOfTrits = _TRITS_IN_UINT_);		
		TritSet(const TritSet& orig);					
		~TritSet();										

		uint getNum (size_t ind) {
			return this->trits[ind];
		}

		size_t capacity() const;						
		size_t getNumOfTrits() const;					

		size_t cardinality(Trit val) const;				
		std::unordered_map
		<Trit, int, std::hash<int>> cardinality() const;
		size_t length() const;							

		
		void trim(size_t lastIndex);					
		void shrink();									


		
		TritHelp operator[](size_t tritIndex);
		Trit operator[](size_t tritIndex) const;

		TritSet& operator=(const TritSet& rightSet);	

		TritSet operator&(TritSet& rightSet) const;
		TritSet& operator&=(TritSet& rightSet);

		TritSet operator|(TritSet& rightSet) const;
		TritSet& operator|=(TritSet& rightSet);

		TritSet operator~() const;
	};
}



#endif