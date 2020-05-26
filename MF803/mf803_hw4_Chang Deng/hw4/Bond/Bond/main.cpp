#include "bond.h"

double bond_price(int m, double y) {
	double price;
	price = 100 / pow(1 + y, m);
	return price;
}



enum PositionType {
	Equity = 1,
	Bond = 2,
	Option = 3
};

double calcAverageVal(double x, double y) {
	x = 4.0;
	return 0.5 * (x + y);
}
double calcAverageRef(double & x, double & y) {
	x = 4.0;
	return 0.5 * (x + y);
}

double calcAveragePtr(double * x, double * y) {
	 * x = 5.0;
	return 0.5 * (*x + *y);
}




int main(int argc, const char * argv[]) {
	double x = 5.5;
	double y = 8.2;
	cout << x << " " << y << endl;
	double avgRef = calcAverageRef(x, y);
	cout << x << " " << y << endl;
	double avgVal = calcAverageVal(x, y);
	cout << x << " " << y << endl;
	double avgPtr = calcAveragePtr(&x, &y);
	cout << x << " " << y << endl;
	
	bond b1(1, 0.025);
	bond b2(2, 0.026);
	bond b3(3, 0.027);
	bond b4(5, 0.030);
	bond b5(10, 0.035);
	bond b6(30, 0.04);
	//a
	cout << "a" << endl;
	cout << "price of zero coupon bond with Maturity 1 and Yield 0.025: $";
	cout << b1.cal_bond_price() << endl;
	cout << "price of zero coupon bond with Maturity 2 and Yield 0.026: $";
	cout << b2.cal_bond_price() << endl;
	cout << "price of zero coupon bond with Maturity 3 and Yield 0.027: $";
	cout << b3.cal_bond_price() << endl;
	cout << "price of zero coupon bond with Maturity 5 and Yield 0.030: $";
	cout << b4.cal_bond_price() << endl;
	cout << "price of zero coupon bond with Maturity 10 and Yield 0.035: $";
	cout << b5.cal_bond_price() << endl;
	cout << "price of zero coupon bond with Maturity 30 and Yield 0.040: $";
	cout << b6.cal_bond_price() << endl;
	//b
	cout << "b" << endl;
	cout << "duration of zero coupon bond with Maturity 1 and Yield 0.025: ";
	cout << b1.cal_bond_duration() << endl;
	cout << "duration of zero coupon bond with Maturity 2 and Yield 0.026: ";
	cout << b2.cal_bond_duration() << endl;
	cout << "duration of zero coupon bond with Maturity 3 and Yield 0.027: ";
	cout << b3.cal_bond_duration() << endl;
	cout << "duration of zero coupon bond with Maturity 5 and Yield 0.030: ";
	cout << b4.cal_bond_duration() << endl;
	cout << "duration of zero coupon bond with Maturity 10 and Yield 0.035: ";
	cout << b5.cal_bond_duration() << endl;
	cout << "duration of zero coupon bond with Maturity 30 and Yield 0.040: ";
	cout << b6.cal_bond_duration() << endl;
	//c
	cout << "c" << endl;
	coupon_bond cb1(1, 0.025);
	coupon_bond cb2(2, 0.026);
	coupon_bond cb3(3, 0.027);
	coupon_bond cb4(5, 0.030);
	coupon_bond cb5(10, 0.035);
	coupon_bond cb6(30, 0.040);
	cout << "price of coupon bond with Maturity 1 and Yield 0.025: ";
	cout << cb1.cal_bond_price() << endl;
	cout << "price of coupon bond with Maturity 2 and Yield 0.026: $";
	cout << cb2.cal_bond_price() << endl;
	cout << "price of coupon bond with Maturity 3 and Yield 0.027: $";
	cout << cb3.cal_bond_price() << endl;
	cout << "price of coupon bond with Maturity 5 and Yield 0.030: $";
	cout << cb4.cal_bond_price() << endl;
	cout << "price of coupon bond with Maturity 10 and Yield 0.035: $";
	cout << cb5.cal_bond_price() << endl;
	cout << "price of coupon bond with Maturity 30 and Yield 0.040: $";
	cout << cb6.cal_bond_price() << endl;
	//d
	cout << "d" << endl;
	cout << "duration of coupon bond with Maturity 1 and Yield 0.025: ";
	cout << cb1.cal_bond_duration() << endl;
	cout << "duration of coupon bond with Maturity 2 and Yield 0.026: ";
	cout << cb2.cal_bond_duration() << endl;
	cout << "duration of coupon bond with Maturity 3 and Yield 0.027: ";
	cout << cb3.cal_bond_duration() << endl;
	cout << "duration of coupon bond with Maturity 5 and Yield 0.030: ";
	cout << cb4.cal_bond_duration() << endl;
	cout << "duration of coupon bond with Maturity 10 and Yield 0.035: ";
	cout << cb5.cal_bond_duration() << endl;
	cout << "duration of coupon bond with Maturity 30 and Yield 0.040: ";
	cout << cb6.cal_bond_duration() << endl;
	//e
	cout << "e" << endl;
	cout << "convexity of zero-coupon bond with Maturity 1 and Yield 0.025: ";
	cout << b1.cal_bond_convexity() << endl;
	cout << "convexity of zero-coupon bond with Maturity 2 and Yield 0.026: ";
	cout << b2.cal_bond_convexity() << endl;
	cout << "convexity of zero-coupon bond with Maturity 3 and Yield 0.027: ";
	cout << b3.cal_bond_convexity() << endl;
	cout << "convexity of zero-coupon bond with Maturity 5 and Yield 0.030: ";
	cout << b4.cal_bond_convexity() << endl;
	cout << "convexity of zero-coupon bond with Maturity 10 and Yield 0.035: ";
	cout << b5.cal_bond_convexity() << endl;
	cout << "convexity of zero-coupon bond with Maturity 30 and Yield 0.040: ";
	cout << b6.cal_bond_convexity() << endl;
	cout << "convexity of coupon bond with Maturity 1 and Yield 0.025: ";
	cout << cb1.cal_bond_convexity() << endl;
	cout << "convexity of coupon bond with Maturity 2 and Yield 0.026: ";
	cout << cb2.cal_bond_convexity() << endl;
	cout << "convexity of coupon bond with Maturity 3 and Yield 0.027: ";
	cout << cb3.cal_bond_convexity() << endl;
	cout << "convexity of coupon bond with Maturity 5 and Yield 0.030: ";
	cout << cb4.cal_bond_convexity() << endl;
	cout << "convexity of coupon bond with Maturity 10 and Yield 0.035: ";
	cout << cb5.cal_bond_convexity() << endl;
	cout << "convexity of coupon bond with Maturity 30 and Yield 0.040: ";
	cout << cb6.cal_bond_convexity() << endl;
	//f
	cout << "f" << endl;
	double portfolio_value = b1.cal_bond_price() + b3.cal_bond_price() - 2 * b2.cal_bond_price();
	cout << "the initial value of the portfolio: $";
	cout << portfolio_value << endl;
	//g
	cout << "g" << endl;
	double delta_y = 0.01;
	bond b1_(1, 0.025 + delta_y);
	bond b2_(2, 0.026 + delta_y);
	bond b3_(3, 0.027 + delta_y);
	bond b1__(1, 0.025 - delta_y);
	bond b2__(2, 0.026 - delta_y);
	bond b3__(3, 0.027 - delta_y);
	double portfolio_value_ = b1_.cal_bond_price() + b3_.cal_bond_price() - 2 * b2_.cal_bond_price();
	double portfolio_value__ = b1__.cal_bond_price() + b3__.cal_bond_price() - 2 * b2__.cal_bond_price();
	double portfolio_duration = ((portfolio_value__ - portfolio_value_) / (2 * delta_y)) / portfolio_value;
	cout << "the duration of the portfolio: ";
	cout << portfolio_duration << endl;
	double portfolio_convexity = ((portfolio_value_ + portfolio_value__ - 2 * portfolio_value) / (pow(delta_y, 2))) / portfolio_value;
	cout << "the convexity of the portfolio: ";
	cout << portfolio_convexity << endl;
	//h
	cout << "h" << endl;
	//double delta_value1 = portfolio_value_ - portfolio_value;
	//double delta_value1 = delta_y * portfolio_duration;
	cout << "the value of the portfolio under rates sell off by 100 basis points: $";
	cout << portfolio_value_ << endl;
	//i
	cout << "i" << endl;
	//double delta_value2 = portfolio_value__ - portfolio_value;
	//double up_value = -delta_y * portfolio_duration;
	cout << "the value of the portfolio under rates rally by 100 basis points: $";
	cout << portfolio_value__ << endl;
	//j
	cout << "j" << endl;
	amortizing_bond ab(5, 0.03);
	cout << "the price of 5-year amortizing bond that repays 20% of its principal annually and pays a 3% coupon annually: $";
	cout << ab.cal_bond_price() << endl;
	//
	cout << "the duration of 5-year amortizing bond that repays 20% of its principal annually and pays a 3% coupon annually: ";
	cout << ab.cal_bond_duration() << endl;
}
