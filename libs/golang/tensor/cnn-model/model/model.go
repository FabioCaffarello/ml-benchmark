package cnnmodel

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"log"
)

type ConvNet struct {
	g                  *gorgonia.ExprGraph
	w0, w1, w2, w3, w4 *gorgonia.Node
	d0, d1, d2, d3     float64
	Out                *gorgonia.Node
	PredVal            gorgonia.Value
}

func NewConvNet(g *gorgonia.ExprGraph) *ConvNet {
	w0 := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(32, 3, 5, 5), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w1 := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(64, 32, 5, 5), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w2 := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(128, 64, 5, 5), gorgonia.WithName("w2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w3 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(512, 256), gorgonia.WithName("w3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w4 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(256, 10), gorgonia.WithName("w4"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	return &ConvNet{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
		w3: w3,
		w4: w4,
		d0: 0.2,
		d1: 0.2,
		d2: 0.2,
		d3: 0.55,
	}
}

func (m *ConvNet) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1, m.w2, m.w3, m.w4}
}

func (m *ConvNet) Fwd(x *gorgonia.Node) error {
	var c0, c1, c2, fc *gorgonia.Node
	var a0, a1, a2, a3 *gorgonia.Node
	var p0, p1, p2 *gorgonia.Node
	var l0, l1, l2, l3 *gorgonia.Node

	log.Println("Starting forward pass")

	c0, err := gorgonia.Conv2d(x, m.w0, tensor.Shape{5, 5}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return errors.Wrap(err, "Layer 0 Convolution failed")
	}
	log.Println("Layer 0 Convolution successful")

	a0, err = gorgonia.Rectify(c0)
	if err != nil {
		return errors.Wrap(err, "Layer 0 activation failed")
	}
	log.Println("Layer 0 activation successful")

	p0, err = gorgonia.MaxPool2D(a0, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2})
	if err != nil {
		return errors.Wrap(err, "Layer 0 Maxpooling failed")
	}
	log.Println("Layer 0 Maxpooling successful")

	l0, err = gorgonia.Dropout(p0, m.d0)
	if err != nil {
		return errors.Wrap(err, "Unable to apply a dropout")
	}
	log.Println("Layer 0 Dropout successful")

	c1, err = gorgonia.Conv2d(l0, m.w1, tensor.Shape{5, 5}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return errors.Wrap(err, "Layer 1 Convolution failed")
	}
	log.Println("Layer 1 Convolution successful")

	a1, err = gorgonia.Rectify(c1)
	if err != nil {
		return errors.Wrap(err, "Layer 1 activation failed")
	}
	log.Println("Layer 1 activation successful")

	p1, err = gorgonia.MaxPool2D(a1, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2})
	if err != nil {
		return errors.Wrap(err, "Layer 1 Maxpooling failed")
	}
	log.Println("Layer 1 Maxpooling successful")

	l1, err = gorgonia.Dropout(p1, m.d1)
	if err != nil {
		return errors.Wrap(err, "Unable to apply a dropout to layer 1")
	}
	log.Println("Layer 1 Dropout successful")

	c2, err = gorgonia.Conv2d(l1, m.w2, tensor.Shape{5, 5}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return errors.Wrap(err, "Layer 2 Convolution failed")
	}
	log.Println("Layer 2 Convolution successful")

	a2, err = gorgonia.Rectify(c2)
	if err != nil {
		return errors.Wrap(err, "Layer 2 activation failed")
	}
	log.Println("Layer 2 activation successful")

	p2, err = gorgonia.MaxPool2D(a2, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2})
	if err != nil {
		return errors.Wrap(err, "Layer 2 Maxpooling failed")
	}
	log.Println("Layer 2 Maxpooling successful")

	b, c, h, w := p2.Shape()[0], p2.Shape()[1], p2.Shape()[2], p2.Shape()[3]
	r2, err := gorgonia.Reshape(p2, tensor.Shape{b, c * h * w})
	if err != nil {
		return errors.Wrap(err, "Unable to reshape layer 2")
	}
	log.Println("Layer 2 Reshape successful")

	l2, err = gorgonia.Dropout(r2, m.d2)
	if err != nil {
		return errors.Wrap(err, "Unable to apply a dropout on layer 2")
	}
	log.Println("Layer 2 Dropout successful")

	fc, err = gorgonia.Mul(l2, m.w3)
	if err != nil {
		return errors.Wrap(err, "Unable to multiply l2 and w3")
	}
	log.Println("Layer 3 Multiplication successful")

	a3, err = gorgonia.Rectify(fc)
	if err != nil {
		return errors.Wrap(err, "Unable to activate fc")
	}
	log.Println("Layer 3 activation successful")

	l3, err = gorgonia.Dropout(a3, m.d3)
	if err != nil {
		return errors.Wrap(err, "Unable to apply a dropout on layer 3")
	}
	log.Println("Layer 3 Dropout successful")

	out, err := gorgonia.Mul(l3, m.w4)
	if err != nil {
		return errors.Wrap(err, "Unable to multiply l3 and w4")
	}
	log.Println("Output layer multiplication successful")

	m.Out, err = gorgonia.SoftMax(out)
	if err != nil {
		return errors.Wrap(err, "SoftMax operation failed")
	}
	log.Println("SoftMax operation successful")

	if m.Out == nil {
		return errors.New("SoftMax operation returned nil")
	}

	// Ensure PredVal is allocated
	m.PredVal = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(m.Out.Shape()...))

	vm := gorgonia.NewTapeMachine(m.g)
	defer vm.Close()
	if err = vm.RunAll(); err != nil {
		return errors.Wrap(err, "Failed to run graph")
	}
	log.Println("Graph execution successful")

	// Ensure the value is read from the graph
	gorgonia.Read(m.Out, &m.PredVal)
	log.Printf("m.Out value: %v", m.Out.Value())
	log.Printf("m.PredVal value: %v", m.PredVal)

	if m.PredVal == nil {
		return errors.New("Prediction value is nil after graph execution")
	}

	log.Println("Forward pass completed")
	return nil
}
