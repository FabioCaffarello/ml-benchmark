package main

import (
	"flag"
	"fmt"
	"log"

	"libs/golang/tensor/trainer/cnn_trainer"
)

func main() {
	// Define command-line flags for the file paths, epochs, and batch size
	trainFile := flag.String("trainFile", "/app/data/cifar10_train.npz", "Path to the training dataset file")
	testFile := flag.String("testFile", "/app/data/cifar10_test.npz", "Path to the test dataset file")
	epochs := flag.Int("epochs", 10, "Number of epochs to train the model")
	batchSize := flag.Int("batchSize", 100, "Batch size for training")

	flag.Parse()

	// Print the parameters
	fmt.Printf("Training file: %s\n", *trainFile)
	fmt.Printf("Test file: %s\n", *testFile)
	fmt.Printf("Epochs: %d\n", *epochs)
	fmt.Printf("Batch size: %d\n", *batchSize)

	// Train the model
	log.Println("Starting training...")
	cnntrainer.Train(*trainFile, *testFile, *epochs, *batchSize)
	log.Println("Training completed.")
}
