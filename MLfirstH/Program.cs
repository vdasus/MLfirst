using System;
using Microsoft.ML;
using MLfirstHML.Model.DataModels;

namespace MLfirstH
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                var res = Console.ReadLine();
                Console.WriteLine(ConsumeModel(res));
            }
        }

        public static string ConsumeModel(string inp)
        {
            // Load the model
            MLContext mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load("MLModel.zip", out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Use the code below to add input data
            var input = new ModelInput();
            input.Comment = inp;

            // Try model on sample data
            ModelOutput result = predEngine.Predict(input);
            return $"{result.Prediction} - {result.Score}";
        }
    }
}
