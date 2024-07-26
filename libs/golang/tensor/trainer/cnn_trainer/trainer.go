package cnntrainer

import (
	"fmt"
	"log"
	"time"

	"gopkg.in/cheggaaa/pb.v1"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	cnnmodel "libs/golang/tensor/cnn-model/model"
	"libs/golang/tensor/data/loader"
)

func Train(trainFile, testFile string, epochs, batchSize int) {
	xTrain, yTrain, xTest, yTest := loader.LoadData(trainFile, testFile)

	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(batchSize, 3, 32, 32), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(batchSize, 10), gorgonia.WithName("y"))
	m := cnnmodel.NewConvNet(g)

	// Initialize x and y tensors before calling Fwd
	gorgonia.Let(x, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(batchSize, 3, 32, 32)))
	gorgonia.Let(y, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(batchSize, 10)))

	if err := m.Fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	losses := gorgonia.Must(gorgonia.HadamardProd(gorgonia.Must(gorgonia.Log(m.Out)), y))
	cost := gorgonia.Must(gorgonia.Sum(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))

	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	if _, err := gorgonia.Grad(cost, m.Learnables()...); err != nil {
		log.Fatal(err)
	}

	prog, locMap, _ := gorgonia.Compile(g)
	vm := gorgonia.NewTapeMachine(g, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(m.Learnables()...))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(0.001), gorgonia.WithMomentum(0.9))
	defer vm.Close()

	numExamples := xTrain.Shape()[0]
	batches := numExamples / batchSize
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(80)

	for i := 0; i < epochs; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()

		for b := 0; b < batches; b++ {
			start := b * batchSize
			end := start + batchSize
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			xVal, err := xTrain.Slice(sli{start, end})
			if err != nil {
				log.Fatal("Unable to slice x")
			}

			yVal, err := yTrain.Slice(sli{start, end})
			if err != nil {
				log.Fatal("Unable to slice y")
			}
			if err := xVal.(*tensor.Dense).Reshape(batchSize, 3, 32, 32); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch %d: %v", i, err)
			}

			solver.Step(gorgonia.NodesToValueGrads(m.Learnables()))
			vm.Reset()
			bar.Increment()
		}
		log.Printf("Epoch %d | cost %v", i, costVal)

		// Validation step
		validate(xTest, yTest, batchSize, m)
	}
}

func validate(xTest, yTest tensor.Tensor, batchSize int, model *cnnmodel.ConvNet) {
	numTestExamples := xTest.Shape()[0]
	batches := numTestExamples / batchSize

	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(batchSize, 3, 32, 32), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(batchSize, 10), gorgonia.WithName("y"))
	m := cnnmodel.NewConvNet(g)

	// Initialize x and y tensors before calling Fwd
	gorgonia.Let(x, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(batchSize, 3, 32, 32)))
	gorgonia.Let(y, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(batchSize, 10)))

	if err := m.Fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	prog, locMap, _ := gorgonia.Compile(g)
	vm := gorgonia.NewTapeMachine(g, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(m.Learnables()...))
	defer vm.Close()

	correct := 0
	total := 0

	for b := 0; b < batches; b++ {
		start := b * batchSize
		end := start + batchSize
		if start >= numTestExamples {
			break
		}
		if end > numTestExamples {
			end = numTestExamples
		}

		xVal, err := xTest.Slice(sli{start, end})
		if err != nil {
			log.Fatal("Unable to slice x")
		}

		yVal, err := yTest.Slice(sli{start, end})
		if err != nil {
			log.Fatal("Unable to slice y")
		}
		if err := xVal.(*tensor.Dense).Reshape(batchSize, 3, 32, 32); err != nil {
			log.Fatalf("Unable to reshape %v", err)
		}

		gorgonia.Let(x, xVal)
		gorgonia.Let(y, yVal)
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed during validation: %v", err)
		}

		// Evaluate accuracy
		predictions := m.Out.Value().Data().([]float32)
		labels := yVal.Data().([]float32)
		for i := 0; i < len(predictions); i += 10 {
			predictedLabel := argmax(predictions[i : i+10])
			trueLabel := argmax(labels[i : i+10])
			if predictedLabel == trueLabel {
				correct++
			}
			total++
		}

		vm.Reset()
	}

	accuracy := float64(correct) / float64(total)
	log.Printf("Validation Accuracy: %f", accuracy)
}

func argmax(arr []float32) int {
	maxIdx := 0
	maxVal := arr[0]
	for i, val := range arr {
		if val > maxVal {
			maxIdx = i
			maxVal = val
		}
	}
	return maxIdx
}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }
