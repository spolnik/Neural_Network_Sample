namespace NeuronNetwork.Base.Learning
{
    public interface IUnsupervisedLearning
    {
        double Run(double[] input);
        double RunEpoch(double[][] input);
    }
}