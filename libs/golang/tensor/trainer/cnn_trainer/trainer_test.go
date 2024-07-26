package cnntrainer

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
)

type TrainerSuite struct {
	suite.Suite
}

func TestTrainerSuite(t *testing.T) {
	suite.Run(t, new(TrainerSuite))
}

func (suite *TrainerSuite) SetupTest() {
	// Setup code if needed
}

func (suite *TrainerSuite) TestTrain() {
	// Mock file paths (adjust these to your test data paths)
	trainFile := "../tests/data/cifar10/cifar10_train.npz"
	testFile := "../tests/data/cifar10/cifar10_test.npz"

	epochs := 1
	batchSize := 10

	// Ensure no panic
	assert.NotPanics(suite.T(), func() {
		Train(trainFile, testFile, epochs, batchSize)
	})
}
