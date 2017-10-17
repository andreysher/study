#include "trits.hpp"

using namespace Trits;	

TritSet::TritSet (size_t tritsNum): numOfTrits(tritsNum),
 arrLen((tritsNum / _TRITS_IN_UINT_) + !(!(tritsNum % _TRITS_IN_UINT_))) {
	this->trits = new uint[arrLen];

	// new íå î÷èùàåò
	for (size_t ind = 0; ind < arrLen; ind++)
		this->trits[ind] = _UNKNOWN_BLOCK_;
}

TritSet::TritSet(const TritSet& orig) : arrLen(orig.arrLen), numOfTrits(orig.numOfTrits) {
	this->trits = new uint[this->arrLen];

	for (size_t ind = 0; ind < this->arrLen; ind++)
		this->trits[ind] = orig.trits[ind];
}


TritSet& TritSet::operator=(const TritSet& rightSet) {
	if (this == &rightSet)
		return *this;

	this->arrLen = rightSet.arrLen;
	this->numOfTrits = rightSet.numOfTrits;

	this->trits = new uint[this->arrLen];

	for (size_t ind = 0; ind < this->arrLen; ind++)
		this->trits[ind] = rightSet.trits[ind];
	
	return *this;
}

TritSet::~TritSet() {
	delete[] this->trits;
}

TritSet::TritHelp::TritHelp(TritSet *parent, size_t index) : parent(parent), arrIndex(index / _TRITS_IN_UINT_), tritIndex(index % _TRITS_IN_UINT_) {
	this->trit = this->getTrit(index);
}

TritSet::TritHelp::~TritHelp() {
	this->parent = nullptr;
}

Trit TritSet::TritHelp::operator=(Trit val) {
	this->setTrit(val);
	return val;
}

Trit TritSet::TritHelp::operator=(TritHelp& rp) {
	this->setTrit(rp.trit);
	return this->trit;
}

TritSet::TritHelp::operator Trit() {
	return this->trit;
}

Trit TritSet::TritHelp::getTrit(size_t index) {
	uint needNum;

	if (index > parent->numOfTrits)
		return Unknown;

	needNum = parent->trits[this->arrIndex];

	for (ushort shift = 0; shift < this->tritIndex; shift++)
		needNum >>= 2;

	needNum &= _TRUE_;
	switch (needNum)
	{
	case _FALSE_:
		return False;

	case _UNKNOWN_: default:
		return Unknown;

	case _TRUE_:
		return True;
	}
}

void TritSet::TritHelp::setTrit(Trit val) {
	uint changeNum;
	uint killer = _TRUE_;
	uint *tmp;

	if (this->arrIndex >= this->parent->arrLen) {
		if (val == Unknown)
			return;

		// Óâåëè÷åíèå ïàìÿòè
		tmp = new uint[this->arrIndex + 1];

		delete[] this->parent->trits;
		this->parent->trits = tmp;
		this->parent->arrLen = this->arrIndex + 1;
		this->parent->numOfTrits = this->parent->arrLen * _TRITS_IN_UINT_;
	}

	switch (val) {
	case False:
		changeNum = _FALSE_;
		break;

	case Unknown:
		changeNum = _UNKNOWN_;
		break;

	case True:
		changeNum = _TRUE_;
		break;
	}

	for (ushort shift = 0; shift < this->tritIndex; shift++) {
		changeNum <<= 2;
		killer <<= 2;
	}

	parent->trits[this->arrIndex] &= ~killer;
	parent->trits[this->arrIndex] |= changeNum;
}

TritSet::TritHelp TritSet::operator[](size_t tritIndex) {
	return TritHelp(this, tritIndex);
}

Trit TritSet::operator[](size_t tritIndex) const {
	uint needNum;
	size_t index = tritIndex % _TRITS_IN_UINT_;
	size_t arrIndex = tritIndex / _TRITS_IN_UINT_;

	if (tritIndex > this->numOfTrits)
		return Unknown;

	needNum = this->trits[arrIndex];

	for (ushort shift = 0; shift < index; shift++)
		needNum >>= 2;

	needNum &= _TRUE_;
	switch (needNum)
	{
	case _FALSE_:
		return False;

	case _UNKNOWN_: default:
		return Unknown;

	case _TRUE_:
		return True;
	}
}

TritSet TritSet::operator&(TritSet& rightSet) const {
	TritSet res(*this);
	res &= rightSet;
	return res;
}

TritSet& TritSet::operator&=(TritSet& rightSet) {
	size_t maxLen = (this->arrLen > rightSet.arrLen) ? this->arrLen : rightSet.arrLen;
	size_t minLen = (this->arrLen < rightSet.arrLen) ? this->arrLen : rightSet.arrLen;
	uint *bigger = (this->arrLen > rightSet.arrLen) ? this->trits : (&rightSet)->trits;
	uint *mayArr;

	if (maxLen > this->arrLen) {
		mayArr = new uint[maxLen];

		for (size_t ind = 0; ind < this->arrLen; ind++)
			mayArr[ind] = this->trits[ind];

		for (size_t ind = this->arrLen; ind < maxLen; ind++)
			mayArr[ind] = _UNKNOWN_BLOCK_;

		delete[] this->trits;
		this->trits = mayArr;
	}

	for (size_t ind = 0; ind < minLen; ind++)
		this->trits[ind] = this->trits[ind] & rightSet.trits[ind];

	for (size_t ind = minLen; ind < maxLen; ind++)
		this->trits[ind] = _UNKNOWN_BLOCK_ & bigger[ind];

	return *this;
}

TritSet TritSet::operator|(TritSet& rightSet) const {
	TritSet res(*this);
	res |= rightSet;
	return res;
}

TritSet& TritSet::operator|=(TritSet& rightSet) {
	size_t maxLen = (this->arrLen > rightSet.arrLen) ? this->arrLen : rightSet.arrLen;
	size_t minLen = (this->arrLen < rightSet.arrLen) ? this->arrLen : rightSet.arrLen;
	uint *bigger = (this->arrLen > rightSet.arrLen) ? this->trits : (&rightSet)->trits;
	uint *mayArr;

	if (maxLen > this->arrLen) {
		mayArr = new uint[maxLen];

		for (size_t ind = 0; ind < this->arrLen; ind++)
			mayArr[ind] = this->trits[ind];

		for (size_t ind = this->arrLen; ind < maxLen; ind++)
			mayArr[ind] = _UNKNOWN_BLOCK_;
		
		delete[] this->trits;
		this->trits = mayArr;
	}

	for (size_t ind = 0; ind < minLen; ind++)
		this->trits[ind] = this->trits[ind] | rightSet.trits[ind];

	for (size_t ind = minLen; ind < maxLen; ind++)
		this->trits[ind] = _UNKNOWN_BLOCK_ | bigger[ind];

	return *this;
}

TritSet TritSet::operator~() const {
	Trit curTrit;
	TritSet res = *this;

	for (size_t ind = 0; ind < this->numOfTrits; ind++) {
		curTrit = this->operator[](ind);

		if (curTrit == Unknown)
			continue;

		if (curTrit == False)
			res.operator[](ind).setTrit(True);
		else
			res.operator[](ind).setTrit(False);
	}

	return res;
}

void TritSet::trim(size_t lastIndex) {
	size_t arrLen;
	ushort tritIndex;
	uint *res;

	if (lastIndex > this->numOfTrits)
		return;

	tritIndex = lastIndex % _TRITS_IN_UINT_;
	arrLen = (lastIndex / _TRITS_IN_UINT_) + !!tritIndex;
	res = new uint[arrLen];

	for (ushort ind = 0; ind < _TRITS_IN_UINT_ - tritIndex; ind++)
		this->operator[](lastIndex + ind) = Unknown;

	for (size_t ind = 0; ind < arrLen; ind++)
		res[ind] = this->trits[ind];

	this->arrLen = arrLen;
	this->numOfTrits = arrLen * _TRITS_IN_UINT_;

	delete[] this->trits;
	this->trits = res;
}

size_t TritSet::length() const {
	size_t res;
	size_t arrInd = 0;
	short tritInd = -1;

	for (size_t ind = 0; ind < this->arrLen; ind++) {
		if (this->trits[ind] != _UNKNOWN_BLOCK_)
			arrInd = ind;
	}

	res = this->trits[arrInd];

	for (ushort ind = 0; ind < _TRITS_IN_UINT_; ind++, res >>= 2) {
		if ((res & _TRUE_) != _UNKNOWN_)
			tritInd = ind;
	}

	if (tritInd == -1)
		return 0;

	return (res = arrInd * _TRITS_IN_UINT_ + tritInd + 1);
}

size_t TritSet::getNumOfTrits() const {
	return this->numOfTrits;
}

size_t TritSet::capacity() const{
	return this->arrLen;
}

size_t TritSet::cardinality(Trit val) const {
	size_t result = 0;
	uint tmpUint;

	for (size_t arrIndex = 0; arrIndex < this->arrLen; arrIndex++) {
		tmpUint = this->trits[arrIndex];

		if (tmpUint == _UNKNOWN_BLOCK_) {
			if (val == Unknown)
				result += _TRITS_IN_UINT_;

			continue;
		}

		for (ushort tritIndex = 0; tritIndex < _TRITS_IN_UINT_; tritIndex++) {
			if (this->operator[](arrIndex*_TRITS_IN_UINT_ + tritIndex) == val)
				result += 1;
		}
	}

	return result;
}

std::unordered_map<Trit, int, std::hash<int>> TritSet::cardinality() const {
	std::unordered_map<Trit, int, std::hash<int>> res = { {True, 0}, {False, 0}, {Unknown, 0} };

	for (size_t ind = 0; ind < this->numOfTrits; ind++) {
		res.at( (this->operator[](ind)) )++;
	}
		
	return res;
}

void TritSet::shrink() {
	this->trim(this->length());
}