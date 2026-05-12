class FinanceCalculator:
    """
    A simple finance calculator for various financial calculations.
    """

    def __init__(self, principal=0.0, rate=0.0, time=0):
        self = self
        self.principal = principal
        self.rate = rate
        self.time = time

    def pva(self,compunding_frequency=0):
        """
        Calculate the present value payment of annuity.
        
        :param principal: Total amount of the loan
        :param rate: Annual interest rate (as a decimal)
        :param time: Time in years
        :return: Monthly payment amount
        """
        monthly_rate = self.rate / compunding_frequency
        number_of_payments = self.time * compunding_frequency
        return self.principal * (1 - (1/(1 + monthly_rate) ** number_of_payments))/ monthly_rate


    def calculate_future_value(self):
        """
        Calculate the future value of an investment.

        :param principal: Initial amount of money invested
        :param rate: Annual interest rate (as a decimal)
        :param time: Time in years
        :return: Future value of the investment
        """
        return self.principal * (1 + self.rate) ** self.time

    def present_value(self,future_value):
        """
        Calculate the present value of a future amount.

        :param future_value: Future amount of money
        :param rate: Annual interest rate (as a decimal)
        :param time: Time in years
        :return: Present value of the future amount
        """
        return future_value / (1 + self.rate) ** self.time

    def npv(self,cash_flows=None):
        """
        Calculate the net present value of a series of cash flows.

        :param rate: Discount rate (as a decimal)
        :param cash_flows: List of cash flows
        :return: Net present value
        """

        if cash_flows is None:
            cash_flows = []
        npv = 0
        for t, cash_flow in enumerate(cash_flows):
            npv += cash_flow / (1 + self.rate) ** t
        return npv - self.principal

def main():
    mydict = {
        "1": "Calculate Future Value",
        "2": "Calculate Present Value",
        "3": "Calculate Net Present Value",
        "4": "Calculate Present Value of Annuity",
        "": "Exit"
    }
    op = input(
        "Welcome to the Finance Calculator! which function would you like to use?\n1: Calculate Future Value\n2: Calculate Present Value\n3: Calculate Net Present Value\n4: Calculate Present Value of Annuity \n(Press Enter to continue): "
               )
    while op in mydict:
        if op is not None:
            op = input(
            "Welcome to the Finance Calculator! which function would you like to use?\n1: Calculate Future Value\n2: Calculate Present Value\n3: Calculate Net Present Value\n4: Calculate Present Value of Annuity \n(Press Enter to continue): "
                   )
        try:
            print(f"You selected: {mydict[op]}")
            if mydict[op] == "Exit":
                print("Exiting the Finance Calculator. Goodbye!")
                break
            principal = float(input("Enter the principal amount: "))
            rate = float(input("Enter the annual interest rate (as a decimal): "))
            time = int(input("Enter the time in years: "))
            calculator = FinanceCalculator(principal, rate, time)

            if op == "1":
                future_value = calculator.calculate_future_value()
                print(f"The future value of the investment is: {round(future_value, 3)}")   
            elif op == "2":     
                future_value = float(input("Enter the future value amount: "))
                present_value = calculator.present_value(future_value)  
                print(f"The present value of the future amount is: {round(present_value,3)}")
            elif op == "3":
                cash_flows = input("Enter the cash flows separated by commas: ")
                cash_flows = [float(cf) for cf in cash_flows.split(",")]
                npv = calculator.npv(cash_flows)
                print(f"The net present value of the cash flows is: {round(npv,3)}")    
            elif op == "4":
                compounding_frequency = int(input("Enter the compoundin frequency (e.g., 12 for monthly or 1 for anually): "))
                pva = calculator.pva(compounding_frequency)
                print(f"The present value payment of annuity is: {round(pva,3)}")
            op = None
            continue
        except Exception as e:
            print(f"Invalid input. Please enter numeric values. Error: {e}")
            continue

if __name__ == "__main__":
    main()