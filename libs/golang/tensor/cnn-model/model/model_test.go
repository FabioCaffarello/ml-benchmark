package cnnmodel

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type ModelSuite struct {
	suite.Suite
	g *gorgonia.ExprGraph
}

func TestModelSuite(t *testing.T) {
	suite.Run(t, new(ModelSuite))
}

func (suite *ModelSuite) SetupTest() {
	suite.g = gorgonia.NewGraph()
}

func (suite *ModelSuite) TestNewConvNet() {
	model := NewConvNet(suite.g)
	assert.NotNil(suite.T(), model)
	assert.NotNil(suite.T(), model.w0)
	assert.NotNil(suite.T(), model.w1)
	assert.NotNil(suite.T(), model.w2)
	assert.NotNil(suite.T(), model.w3)
	assert.NotNil(suite.T(), model.w4)
	assert.Equal(suite.T(), model.d0, 0.2)
	assert.Equal(suite.T(), model.d1, 0.2)
	assert.Equal(suite.T(), model.d2, 0.2)
	assert.Equal(suite.T(), model.d3, 0.55)
}

func (suite *ModelSuite) TestForward() {
	model := NewConvNet(suite.g)
	x := gorgonia.NewTensor(suite.g, tensor.Float32, 4, gorgonia.WithShape(1, 3, 32, 32), gorgonia.WithInit(gorgonia.Zeroes()))
	err := model.Fwd(x)
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), model.Out)
	assert.NotNil(suite.T(), model.PredVal)
}
