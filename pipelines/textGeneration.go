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

// GeneratedOutput holds the generated output text.
type GeneratedOutput struct {
	GeneratedText string
}

// GetOutput
func (t *GeneratedOutput) GetOutput() []any {
	return []any{t.GeneratedText}
}

// NewTextGenerationPipeline initializes a new text generation pipeline.
func NewTextGenerationPipeline(config pipelineBackends.PipelineConfig[*TextGenerationPipeline], s *options.Options, model *pipelineBackends.Model) (*TextGenerationPipeline, error) {
	basePipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &TextGenerationPipeline{
		BasePipeline: basePipeline,
	}

	// Apply functional options
	for _, opt := range config.Options {
		opt(pipeline)
	}

	// Read configuration, such as prefix
	configPath := util.PathJoinSafe(model.Path, "config.json")
	configBytes, err := util.ReadFileBytes(configPath)
	if err != nil {
		return nil, err
	}

	var pipelineConfig TextGenerationPipelineConfig
	if err := jsoniter.Unmarshal(configBytes, &pipelineConfig); err != nil {
		return nil, err
	}

	pipeline.Prefix = pipelineConfig.Prefix

	return pipeline, nil
}

// Preprocess prepares the input text for generation.
func (p *TextGenerationPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []string) error {
	start := time.Now()

	for i, input := range inputs {
		inputs[i] = p.Prefix + input
	}

	err := pipelineBackends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	return err
}

// GetModel
func (p *TextGenerationPipeline) GetModel() *pipelineBackends.Model {
	return p.BasePipeline.Model
}

// GetMetadata
func (p *TextGenerationPipeline) GetMetadata() pipelineBackends.PipelineMetadata {

}

// Validate
func (p *TextGenerationPipeline) Validate() error {
	return nil
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

// Forward performs the model inference step.
func (p *TextGenerationPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()
	err := pipelineBackends.RunSessionOnBatch(batch, p.BasePipeline)
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return err
}

// Postprocess extracts generated text from the model's output.
func (p *TextGenerationPipeline) Postprocess(batch *pipelineBackends.PipelineBatch) ([]GeneratedOutput, error) {
	output := batch.OutputValues[0]
	if len(output.Result2D) == 0 {
		return nil, errors.New("output is not 2D, expected batch size x sequence length")
	}

	results := make([]GeneratedOutput, len(batch.Input))
	for i, tokens := range output.Result2D {
		text := p.Model.Tokenizer.Decode(tokens)
		results[i] = GeneratedOutput{
			GeneratedText: text,
		}
	}

	return results, nil
}

func (p *TextGenerationPipeline) Run(inputs []string) (GeneratedOutput, error) {

	return p.RunPipeline(inputs)
}

// RunPipeline runs the text generation pipeline on a batch of inputs.
func (p *TextGenerationPipeline) RunPipeline(inputs []string) (GeneratedOutput, error) {
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
