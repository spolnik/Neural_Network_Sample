using System.Collections.Generic;

namespace NeuronNetwork.Base.Layers
{
    public interface ILayer
    {
        int InputsCount { get; }
		double Alpha {set;get;}
		int Dim{set; get;}
        INeuronsCollection Neurons { get; }
        IList<double> Outputs { get; }
        IList<double> Compute(IList<double> inputs);
		void Learn(int numberOfEpoch, List<List<double>> input);
		void Learn1D(int R, int index, List<double> wektorUczacy);

        int GetWinner();
    }
}