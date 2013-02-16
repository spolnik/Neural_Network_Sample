﻿namespace NeuronNetwork.Base.Learning
{
    public interface ISupervisedLearning
    {
        double Run(double[] input, double[] output);
        double RunEpoch(double[][] input, double[][] output);
    }
}