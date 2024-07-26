package loader

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
	"gorgonia.org/tensor"
)

type DataLoaderSuite struct {
	suite.Suite
}

func TestDataLoaderSuite(t *testing.T) {
	suite.Run(t, new(DataLoaderSuite))
}

func (suite *DataLoaderSuite) TestLoadData() {
	// Mock file paths (adjust these to your test data paths)
	trainFile := "../tests/data/cifar10/cifar10_train.npz"
	testFile := "../tests/data/cifar10/cifar10_test.npz"

	xTrain, yTrain, xTest, yTest := LoadData(trainFile, testFile)

	assert.NotNil(suite.T(), xTrain)
	assert.NotNil(suite.T(), yTrain)
	assert.NotNil(suite.T(), xTest)
	assert.NotNil(suite.T(), yTest)

	assert.Equal(suite.T(), len(xTrain.Shape()), 4)
	assert.Equal(suite.T(), len(yTrain.Shape()), 2)

	assert.Equal(suite.T(), len(xTest.Shape()), 4)
	assert.Equal(suite.T(), len(yTest.Shape()), 2)

	assert.Equal(suite.T(), xTrain.Dtype(), tensor.Float32)
	assert.Equal(suite.T(), yTrain.Dtype(), tensor.Float32)
	assert.Equal(suite.T(), xTest.Dtype(), tensor.Float32)
	assert.Equal(suite.T(), yTest.Dtype(), tensor.Float32)
}

func (suite *DataLoaderSuite) TestLoadNPY() {
	// Mock file path (adjust this to your test data path)
	npzFile := "../tests/data/cifar10/cifar10_train.npz"

	x, y := loadNPZ(npzFile)

	assert.NotNil(suite.T(), x)
	assert.NotNil(suite.T(), y)

	assert.Equal(suite.T(), len(x.Shape()), 4)
	assert.Equal(suite.T(), len(y.Shape()), 1)

	assert.Equal(suite.T(), x.Dtype(), tensor.Float32)
	assert.Equal(suite.T(), y.Dtype(), tensor.Uint8)
}
