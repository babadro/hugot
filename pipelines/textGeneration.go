package pipelines

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/util"

	jsoniter "github.com/json-iterator/go"
)

// TextGenerationPipeline represents a text generation pipeline.
type TextGenerationPipeline struct {
	*pipelineBackends.BasePipeline
	Prefix string
}

// TextGenerationPipelineConfig holds configuration for the pipeline.
type TextGenerationPipelineConfig struct {
	Prefix string `json:"prefix"`
}

// TextGenerationOutput holds the generated output text.
type TextGenerationOutput struct {
	GeneratedText string
}

// GetOutput
func (t TextGenerationOutput) GetOutput() []any {
	return []any{t.GeneratedText}
}

// NewTextGenerationPipeline initializes a new text generation pipeline.
func NewTextGenerationPipeline(config pipelineBackends.PipelineConfig[*TextGenerationPipeline], s *options.Options, model *pipelineBackends.Model) (*TextGenerationPipeline, error) {
	defaultPipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &TextGenerationPipeline{BasePipeline: defaultPipeline}

	// Apply functional options
	for _, o := range config.Options {
		o(pipeline)
	}

	return pipeline, nil
}
// GetModel
func (p *TextGenerationPipeline) GetModel() *pipelineBackends.Model {
	return p.BasePipeline.Model
}

// GetMetadata
func (p *TextGenerationPipeline) GetMetadata() pipelineBackends.PipelineMetadata {
	return pipelineBackends.PipelineMetadata{
	OutputsInfo: []pipelineBackends.OutputInfo{
		{
			Name: p.Model.OutputsMeta[0].Name,
			Dimensions: p.Model.OutputsMeta[0].Dimensions,
		},
	},
	}
}


// GetStats
func (p *TextGenerationPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("Tokenizer: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.Model.Tokenizer.TokenizerTimings.TotalNS),
			p.Model.Tokenizer.TokenizerTimings.NumCalls,
			time.Duration(float64(p.Model.Tokenizer.TokenizerTimings.TotalNS)/math.Max(1, float64(p.Model.Tokenizer.TokenizerTimings.NumCalls)))),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls))),
	}
}

// Validate
func (p *TextGenerationPipeline) Validate() error {
	// todo
	return nil
}



// Preprocess prepares the input text for generation.
func (p *TextGenerationPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []string) error {
	start := time.Now()
	pipelineBackends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	err := pipelineBackends.CreateInputTensors(batch, p.Model.InputsMeta, p.Runtime)
	return err
}


// Forward performs the model inference step.
func (p *TextGenerationPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()
	err := pipelineBackends.RunSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

// Postprocess extracts generated text from the model's output.
func (p *TextGenerationPipeline) Postprocess(batch *pipelineBackends.PipelineBatch) ([]TextGenerationOutput, error) {

}

func (p *TextGenerationPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline runs the text generation pipeline on a batch of inputs.
func (p *TextGenerationPipeline) RunPipeline(inputs []string) (TextGenerationOutput, error) {
	batch := pipelineBackends.NewBatch()
	defer batch.Destroy()

	if err := p.Preprocess(batch, inputs); err != nil {
		return nil, err
	}

	if err := p.Forward(batch); err != nil {
		return nil, err
	}

	return p.Postprocess(batch)
}
