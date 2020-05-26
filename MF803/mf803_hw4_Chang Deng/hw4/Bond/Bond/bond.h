#ifndef BOND_H
#define BOND_H
#include <iostream>
using namespace std;


class bond {
public:

	bond();
	bond(int maturity_, double yield_);
	bond(const bond &source);
	double cal_bond_price(int principle = 100);
	double cal_bond_duration(double dy = 0.0001);
	double cal_bond_convexity(double dy = 0.0001);
	int maturity;
	double yield;

};

class coupon_bond :public bond {
public:

	coupon_bond():bond(){};
	coupon_bond(int maturity_, double yield_, double annual_interest_rate_=0.03) :bond(maturity_, yield_) { annual_interest_rate = annual_interest_rate_; };
	coupon_bond(const coupon_bond &source);
	double cal_bond_price(int principle = 100);
	double cal_bond_duration(double dy = 0.0001);
	double cal_bond_convexity(double dy = 0.0001);
	double annual_interest_rate;
};

class amortizing_bond :public coupon_bond {
public:

	amortizing_bond() :coupon_bond() { repay_rate = 0; };
	amortizing_bond(int maturity_, double yield_, double annual_interest_rate_=0.03,double repay_rate_=0.2) :coupon_bond(maturity_, yield_, annual_interest_rate_) {repay_rate = repay_rate_;};
	amortizing_bond(const amortizing_bond &source);
	double cal_bond_price(int principle = 100);
	double cal_bond_duration(double dy = 0.0001);
	double cal_bond_convexity(double dy = 0.0001);
	double repay_rate;
};
#endif
