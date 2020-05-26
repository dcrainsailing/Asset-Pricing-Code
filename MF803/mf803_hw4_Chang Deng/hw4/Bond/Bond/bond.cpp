#include "bond.h"
#include <iostream>
#include <assert.h>

bond::bond() {
	maturity = 0;
	yield = 0.0;
};

bond::bond(int maturity_, double yield_) {
	maturity = maturity_;
	yield = yield_;
};

bond::bond(const bond &source) {
	maturity = source.maturity;
	yield = source.yield;
};

double bond::cal_bond_price(int principle) {
	return principle / pow(1 + yield, maturity);
};

double bond::cal_bond_duration(double dy){
	assert(yield - dy);
	bond b1(maturity, (yield + dy));
	bond b0(maturity, yield);
	bond b2(maturity, (yield - dy));
	double price1 = b1.cal_bond_price();
	double price0 = b0.cal_bond_price();
	double price2 = b2.cal_bond_price();
	double duration = (price2 - price1) / (2*dy)/price0;
	return duration;
};

double bond::cal_bond_convexity(double dy) {
	assert(yield - 2*dy);
	bond b1(maturity, (yield + dy));
	bond b0(maturity, yield);
	bond b2(maturity, (yield - dy));
	double price1 = b1.cal_bond_price();
	double price0 = b0.cal_bond_price();
	double price2 = b2.cal_bond_price();
	//cout << price1 << " " << price2 << " " << price0 << endl;
	//cout << (price1 + price2 - 2 * price0) << endl;
	double convexity = ((price1 + price2 - 2 * price0) / (pow(dy, 2)))/ price0;
	return convexity;
};


coupon_bond::coupon_bond(const coupon_bond &source) {
	maturity = source.maturity;
	yield = source.yield;
	annual_interest_rate = source.annual_interest_rate;
};

double coupon_bond::cal_bond_price(int principle) {
	double price = 0;
	double annual_interest = annual_interest_rate * principle;
	for (int i = 1; i <= maturity; i++) {
		price += annual_interest / pow(1 + yield, i);
	}
	price += principle / pow(1 + yield, maturity);
	return price;
}

double coupon_bond::cal_bond_duration(double dy) {
	assert(yield - dy);
	coupon_bond cb1(maturity, (yield + dy));
	coupon_bond cb0(maturity, yield);
	coupon_bond cb2(maturity, (yield - dy));
	double price1 = cb1.cal_bond_price();
	double price0 = cb0.cal_bond_price();
	double price2 = cb2.cal_bond_price();
	double duration = ((price2 - price1) / (2 * dy))/ price0;
	return duration;
};

double coupon_bond::cal_bond_convexity(double dy) {
	assert(yield - 2 * dy);
	coupon_bond cb1(maturity, (yield + dy));
	coupon_bond cb0(maturity, yield);
	coupon_bond cb2(maturity, (yield - dy));
	double price1 = cb1.cal_bond_price();
	double price0 = cb0.cal_bond_price();
	double price2 = cb2.cal_bond_price();
	double convexity = ((price1 + price2 - 2 * price0) / (pow(dy, 2)))/ price0;
	return convexity;
};

amortizing_bond::amortizing_bond(const amortizing_bond &source) {
	maturity = source.maturity;
	yield = source.yield;
	annual_interest_rate = source.annual_interest_rate;
	repay_rate = source.repay_rate;
};

double amortizing_bond::cal_bond_price(int principle) {
	double price = 0;
	double annual_interest;
	double annual_repay; 
	annual_repay = repay_rate * principle;
	for (int i = 1; i <= maturity; i++) {
		annual_interest = annual_interest_rate * principle;
		//cout << "the cashflow of 5-year amortizing bond that repays 20% of its principal annually and pays a 3% coupon annually: $";
		//cout << (annual_interest + annual_repay) / pow(1 + yield, i) << endl;
		price += (annual_interest+ annual_repay) / pow(1 + yield, i);
		principle = principle - annual_repay;
	}
	return price;
}

double amortizing_bond::cal_bond_duration(double dy) {
	assert(yield - dy);
	amortizing_bond ab1(maturity, (yield + dy));
	amortizing_bond ab0(maturity, yield);
	amortizing_bond ab2(maturity, (yield - dy));
	double price1 = ab1.cal_bond_price();
	double price0 = ab0.cal_bond_price();
	double price2 = ab2.cal_bond_price();
	double duration = ((price2 - price1) / (2 * dy)) / price0;
	return duration;
};

double amortizing_bond::cal_bond_convexity(double dy) {
	assert(yield - 2 * dy);
	amortizing_bond ab1(maturity, (yield + dy));
	amortizing_bond ab0(maturity, yield);
	amortizing_bond ab2(maturity, (yield - dy));
	double price1 = ab1.cal_bond_price();
	double price0 = ab0.cal_bond_price();
	double price2 = ab2.cal_bond_price();
	double convexity = ((price1 + price2 - 2 * price0) / (pow(dy, 2))) / price0;
	return convexity;
};