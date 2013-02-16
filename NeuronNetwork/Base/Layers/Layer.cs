using System;
using System.Collections.Generic;
using NeuronNetwork.Base.Functions;
using NeuronNetwork.Base.Neurons;

namespace NeuronNetwork.Base.Layers
{
    public class Layer : ILayer
    {
        public Layer(int neuronsCount, int inputsCount, IActivationFunction function)
        {
            this.InputsCount = Math.Max(1, inputsCount);
            this.Neurons = new NeuronsCollection(Math.Max(1, neuronsCount), this.InputsCount, function);
            this.Outputs = new OutputsCollection(Math.Max(1, neuronsCount));
        }
		
		
			public void Learn(int numberOfEpoch, List<List<double>> input){
			
			int R = 1;
			
			for(int epoka = 0; epoka < numberOfEpoch; epoka++ , Alpha *= 0.999 ){
				
				//Console.WriteLine("alfa: " + alpha);
			
				//TODO: refactor!
				double min = Inf;
				double sum;		
				

				INeuronsCollection neurony = this.Neurons;

				foreach (List<double> wektorUczacy in input)
                {
					int indexWybranegoNeuronu = 0;
					int ind = 0;		
					min = Inf;		
					foreach (INeuron n in neurony)
                    {
						sum = 0;
						int i = 0;
						foreach (double val in n.Weights)
                        {
							sum += 	(val - wektorUczacy[i]) * (val - wektorUczacy[i]);
							i++;
							//Console.WriteLine(sum);
						}
						
						if(sum < min)
                        {
							min = sum;
							indexWybranegoNeuronu = ind;
						}
						
						ind++;

					}
					//Console.WriteLine("wybrany neuron:" + index_wybranego_neuronu);
					
					//update wag zwycieskiego neuronu
					//w zaleznosci od sasiedztwa
					
					switch (this.Dim)
					{
					    case 0:
					        {
					            int index = 0;
					            for (int i = 0; i < neurony[indexWybranegoNeuronu].Weights.Count; i++)
					            {
					                double val = neurony[indexWybranegoNeuronu].Weights[i];
					                neurony[indexWybranegoNeuronu].Weights[index] = val + this.Alpha*(wektorUczacy[index] - val);
					                index++;
					            }
					        }
					        break;
					    case 1:
					        {
								Learn1D(R, indexWybranegoNeuronu,wektorUczacy);
						    }
					        break;
					    case 2:
					        {
					     		Learn2D(R, indexWybranegoNeuronu,wektorUczacy);
					        }
					        break;
					}
				}
			}
		}
		
		//odleglosc wg metryki taksowkowej
		public void Learn2D(int R, int indexWybranegoNeuronu, List<double> wektorUczacy){	
			
			//TODO: co jesli nie kwadrat a prostokat?
			INeuronsCollection neurony = this.Neurons;
			int rowcount = (int)Math.Sqrt(neurony.Count);
			
			//wspolrzedne uczonego neuronu 
			int row = indexWybranegoNeuronu / rowcount;
			int col = indexWybranegoNeuronu % rowcount;
			
			int current_x, current_y, index;
			
			for(int i = 0; i < neurony.Count; i++)
			{
				current_x = i / rowcount;
				current_y = i % rowcount;
				
				//sprawdzamy czy sasiednie
				if ( Math.Abs(row - current_x) + Math.Abs(col - current_y) <= R){
					index = 0;
					
					foreach (double val in neurony[i].Weights)
					{
						neurony[i].Weights[index] = val + this.Alpha *  (wektorUczacy[index] - val);
						index++;
					}
				}	
			}
		}
		
		public void Learn1D(int R, int indexWybranegoNeuronu, List<double> wektorUczacy){
			int index = 0;
			INeuronsCollection neurony = this.Neurons;	
			
			//uczymy neurony na lewo
			for(int r = 1; r <= R; r++){	
				
				if (indexWybranegoNeuronu - r >= 0)
					foreach (double val in neurony[indexWybranegoNeuronu - r].Weights)
					{
						neurony[indexWybranegoNeuronu - r].Weights[index] = val + this.Alpha*(wektorUczacy[index] - val);
						index++;
					}
			}
			
				//uczymy neurony w srodku
				index = 0;
				foreach (double val in neurony[indexWybranegoNeuronu].Weights)
				{
					neurony[indexWybranegoNeuronu].Weights[index] = val + this.Alpha *  (wektorUczacy[index] - val);
				    index++;
				}
					
			//uczymy neurony na prawo
			for(int r = 1; r <= R; r++){
				
				if(indexWybranegoNeuronu + r < neurony.Count)
				{
					index = 0;
					foreach (double val in neurony[indexWybranegoNeuronu].Weights)
					{
						neurony[indexWybranegoNeuronu + r].Weights[index] = val + this.Alpha *  (wektorUczacy[index] - val);
						index++;
					}
				}
			}
			
		}
		
		
        #region ILayer Members

        /// <summary>
        /// Get winner neuron
        /// </summary>
        /// 
        /// <returns>Index of the winner neuron</returns>
        /// 
        /// <remarks>The method returns index of the neuron, which weights have
        /// the minimum distance from network's input.</remarks>
        /// 
        public int GetWinner()
        {
            double min = this.Outputs[0];
            int minIndex = 0;

            for (int i = 1; i < this.Outputs.Count; i++)
            {
                if (this.Outputs[i] <= min)
                    continue;

                min = this.Outputs[i];
                minIndex = i;
            }

            return minIndex;
        }

		public double Alpha {get;  set;}
		
        public int InputsCount { get; private set; }

		public int Dim{get; set;}
		
        public INeuronsCollection Neurons { get; private set; }

        public IList<double> Outputs { get; private set; }
		
		private const int Inf = 1000000;

        public IList<double> Compute(IList<double> inputs)
        {
            for (int i = 0; i < this.Neurons.Count; i++)
            {
                this.Outputs[i] = this.Neurons[i].Compute(inputs);
            }

            return this.Outputs;
        }

        #endregion
    }
}