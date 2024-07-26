package loader

import (
	"log"
	"os"

	"github.com/sbinet/npyio/npz"
	"gorgonia.org/tensor"
)

func LoadData(trainFile, testFile string) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	// Load training data
	xTrain, yTrain := loadNPZ(trainFile)
	xTest, yTest := loadNPZ(testFile)

	// Normalize the data
	normalize(xTrain)
	normalize(xTest)

	// One-hot encode labels
	yTrain = oneHotEncode(yTrain, 10)
	yTest = oneHotEncode(yTest, 10)

	return xTrain, yTrain, xTest, yTest
}

func loadNPZ(file string) (tensor.Tensor, tensor.Tensor) {
	r, err := os.Open(file)
	if err != nil {
		log.Fatalf("Failed to open file: %v", err)
	}
	defer r.Close()

	// Get the file size
	fi, err := r.Stat()
	if err != nil {
		log.Fatalf("Failed to get file info: %v", err)
	}

	npzReader, err := npz.NewReader(r, fi.Size())
	if err != nil {
		log.Fatalf("Failed to create NPZ reader: %v", err)
	}

	log.Println("Available keys in the NPZ file:")
	for _, name := range npzReader.Keys() {
		log.Printf("key name  %s\n", name)
	}

	var xData []float32
	var yData []uint8
	var xShape, yShape []int

	err = npzReader.Read("x.npy", &xData)
	if err != nil {
		log.Fatalf("Failed to read x data: %v", err)
	}

	err = npzReader.Read("y.npy", &yData)
	if err != nil {
		log.Fatalf("Failed to read y data: %v", err)
	}

	log.Printf("x data type: %T, length: %d\n", xData, len(xData))
	log.Printf("y data type: %T, length: %d\n", yData, len(yData))

	xShape = []int{len(xData) / (32 * 32 * 3), 3, 32, 32}
	yShape = []int{len(yData)}

	log.Printf("x shape: %v\n", xShape)
	log.Printf("y shape: %v\n", yShape)

	x := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(xShape...), tensor.WithBacking(xData))
	y := tensor.New(tensor.Of(tensor.Uint8), tensor.WithShape(yShape...), tensor.WithBacking(yData))

	return x, y
}

func normalize(x tensor.Tensor) {
	data := x.Data().([]float32)
	for i := range data {
		data[i] /= 255.0
	}
}

func oneHotEncode(y tensor.Tensor, numClasses int) tensor.Tensor {
	data := y.Data().([]uint8)
	oneHot := make([]float32, len(data)*numClasses)
	for i, label := range data {
		oneHot[i*numClasses+int(label)] = 1.0
	}
	return tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(len(data), numClasses), tensor.WithBacking(oneHot))
}
