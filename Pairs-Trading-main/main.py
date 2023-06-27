# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2020 QuantConnect Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Model import Model

class ModulatedMultidimensionalAtmosphericScrubbers(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2015, 8, 15)  # Set Start Date
        self.SetEndDate(2020, 8, 15)
        self.SetCash(100000)  # Set Strategy Cash
        self.SetBenchmark('SPY')
        
        self.A = self.AddEquity('GLD', Resolution.Daily).Symbol
        self.B = self.AddEquity('SLV', Resolution.Daily).Symbol
        self.SetWarmup(200)
        self.model = Model()
        
        # retrain our model periodically
        self.Train(self.DateRules.MonthStart('GLD'), self.TimeRules.Midnight, self.TrainModel)
        self.months = 0
        
    def OnData(self, data):
        self.model.Update(self.Time, data[self.A].Close, data[self.B].Close)
        
        if self.IsWarmingUp:
            return
        
        if not self.model.IsReady:
            return
        
        # if we aren't holding the portfolio and our model tells us to buy
        #   the portfolio, we buy the portfolio
        if not self.Portfolio.Invested and self.model.IsEnter():
            self.SetHoldings(self.A, 1) 
            self.SetHoldings(self.B, -self.model.AllocationB())
        # if we are holding the portfolio and our model tells us to sell
        #   the portfolio, we liquidate our holdings
        elif self.Portfolio.Invested and self.model.IsExit():
            self.Liquidate()
        
    def TrainModel(self):
        if not self.model.Ready2Train:
            return
        
        self.months += 1
        
        # only retrain every 7 months
        if self.months % 7 != 1:
            return
    
        self.model.Train()
        
        if not self.model.IsReady:
            self.Liquidate()
            return
            
        theta, mu, sigma = self.model.get_coefficients()
        
        self.Log('θ: ' + str(round(theta, 2)) + ' μ: ' + str(round(mu, 2)) + ' σ: ' + str(round(sigma, 2)))