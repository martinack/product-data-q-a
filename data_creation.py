"""Data generation script for creating synthetic product data and Q&A pairs using LLM."""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Any

import ollama
from tqdm import tqdm


@dataclass
class GenerationConfig:
    """Configuration for data generation."""
    model_name: str = "gpt-oss:20b"
    number_of_products: int = 3
    output_dir: Path = Path(".")
    log_dir: Path = Path("logs")
    products_prompt_file: Path = Path("products_system.prompt")
    questions_prompt_file: Path = Path("questions_system.prompt")
    temperature: float = 0.2
    base_seed: int = 42
    think_mode: str = "medium"

    def validate(self) -> None:
        """Validate configuration."""
        if not self.products_prompt_file.exists():
            raise ValueError(f"Products prompt file not found: {self.products_prompt_file}")
        if not self.questions_prompt_file.exists():
            raise ValueError(f"Questions prompt file not found: {self.questions_prompt_file}")
        if self.number_of_products <= 0:
            raise ValueError("Number of products must be positive")
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Statistics:
    """Track statistics for generated products."""
    product_count: int = 0
    domains: Dict[str, int] = None
    subcategories: Dict[str, int] = None
    units: Dict[str, int] = None
    currencies: Dict[str, int] = None
    colors: Dict[str, int] = None
    materials: Dict[str, int] = None
    capacities: Dict[str, int] = None
    stock_statuses: Dict[str, int] = None
    product_names: List[str] = None
    latest_product: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize defaultdict fields."""
        if self.domains is None:
            self.domains = defaultdict(int)
        if self.subcategories is None:
            self.subcategories = defaultdict(int)
        if self.units is None:
            self.units = defaultdict(int)
        if self.currencies is None:
            self.currencies = defaultdict(int)
        if self.colors is None:
            self.colors = defaultdict(int)
        if self.materials is None:
            self.materials = defaultdict(int)
        if self.capacities is None:
            self.capacities = defaultdict(int)
        if self.stock_statuses is None:
            self.stock_statuses = defaultdict(int)
        if self.product_names is None:
            self.product_names = []


@dataclass
class QAStatistics:
    """Track Q&A generation statistics."""
    factual: int = 0
    multihop: int = 0
    trap: int = 0

    @property
    def total(self) -> int:
        """Return total Q&A count."""
        return self.factual + self.multihop + self.trap

class ConversationLogger:
    """Logger for LLM conversations."""

    def __init__(self, log_dir: Path):
        """
        Initialize conversation logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_conversation(
        self,
        conversation_type: str,
        product_id: Optional[str],
        system_message: str,
        user_message: str,
        response_content: str,
        reasoning: Optional[str],
        start_time: datetime,
        end_time: datetime,
        duration: float,
        seed: int,
        temperature: float,
        model_name: str
    ) -> None:
        """
        Log a complete LLM conversation.

        Args:
            conversation_type: Type of conversation ('product' or 'qa')
            product_id: Product ID (None for product generation)
            system_message: System prompt
            user_message: User prompt
            response_content: LLM response content
            reasoning: LLM reasoning/thinking output
            start_time: Conversation start timestamp
            end_time: Conversation end timestamp
            duration: Duration in seconds
            seed: Random seed used
            temperature: Temperature used
            model_name: Model name used
        """
        # Create log entry
        log_entry = {
            "conversation_type": conversation_type,
            "product_id": product_id,
            "model": model_name,
            "timestamp": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_seconds": round(duration, 3)
            },
            "parameters": {
                "temperature": temperature,
                "seed": seed
            },
            "conversation": {
                "system_message": system_message,
                "user_message": user_message,
                "response": response_content
            },
            "reasoning": reasoning
        }

        # Generate filename
        timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
        if product_id:
            filename = f"{timestamp_str}_{conversation_type}_{product_id}.json"
        else:
            filename = f"{timestamp_str}_{conversation_type}_{seed}.json"

        # Save log file
        log_path = self.log_dir / filename
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)


class LLMClient:
    """Client for interacting with Ollama LLM."""

    def __init__(self, config: GenerationConfig):
        """
        Initialize LLM client.

        Args:
            config: Generation configuration
        """
        self.config = config

    def generate(self, system_message: str, user_message: str, seed: int) -> tuple[Dict[str, Any], datetime, datetime, float]:
        """
        Generate a response from the LLM.

        Args:
            system_message: System prompt
            user_message: User prompt
            seed: Random seed for reproducibility

        Returns:
            Tuple of (response, start_time, end_time, duration)

        Raises:
            Exception: If LLM call fails
        """
        try:
            start_time = datetime.now()
            start_timestamp = time()

            response = ollama.chat(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                options={
                    "temperature": self.config.temperature,
                    "seed": seed
                },
                think=self.config.think_mode,
                format='json'
            )

            end_time = datetime.now()
            duration = time() - start_timestamp

            return response, start_time, end_time, duration
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}") from e


class FileHandler:
    """Handle file operations for generated data."""

    def __init__(self, output_dir: Path):
        """
        Initialize file handler.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir

    def load_prompt(self, prompt_file: Path) -> str:
        """
        Load prompt from file.

        Args:
            prompt_file: Path to prompt file

        Returns:
            Prompt content

        Raises:
            IOError: If file cannot be read
        """
        try:
            return prompt_file.read_text(encoding='utf-8')
        except Exception as e:
            raise IOError(f"Failed to load prompt from {prompt_file}: {str(e)}") from e

    def save_json(self, filename: str, data: Any) -> None:
        """
        Save data as JSON file.

        Args:
            filename: Name of the file
            data: Data to save

        Raises:
            IOError: If file cannot be written
        """
        try:
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"Failed to save JSON to {filename}: {str(e)}") from e

    def save_text(self, filename: str, content: str) -> None:
        """
        Save text content to file.

        Args:
            filename: Name of the file
            content: Text content to save

        Raises:
            IOError: If file cannot be written
        """
        try:
            filepath = self.output_dir / filename
            filepath.write_text(content, encoding='utf-8')
        except Exception as e:
            raise IOError(f"Failed to save text to {filename}: {str(e)}") from e


class StatisticsTracker:
    """Track and update generation statistics."""

    @staticmethod
    def update_product_stats(product: Dict[str, Any], stats: Statistics) -> None:
        """
        Update statistics based on product data.

        Args:
            product: Product data dictionary
            stats: Statistics object to update
        """
        stats.product_count += 1

        # Track basic fields
        if 'category' in product:
            stats.domains[product['category']] += 1
        if 'subcategory' in product:
            stats.subcategories[product['subcategory']] += 1
        if 'name' in product:
            stats.product_names.append(product['name'])

        # Track specs
        specs = product.get('specs', {})
        if 'dimensions' in specs and 'unit' in specs['dimensions']:
            stats.units[f"dimensions_{specs['dimensions']['unit']}"] += 1
        if 'weight' in specs and 'unit' in specs['weight']:
            stats.units[f"weight_{specs['weight']['unit']}"] += 1
        if 'capacity' in specs and specs['capacity'].get('applicable') and 'unit' in specs['capacity']:
            stats.capacities[specs['capacity']['unit']] += 1
        if 'color' in specs:
            stats.colors[specs['color']] += 1
        if 'materials' in specs:
            for material in specs['materials']:
                stats.materials[material] += 1

        # Track price currency
        price = product.get('price', {})
        if 'currency' in price:
            stats.currencies[price['currency']] += 1

        # Track stock status
        if 'stock_status' in product:
            stats.stock_statuses[product['stock_status']] += 1

    @staticmethod
    def update_qa_stats(qa_items: List[Dict[str, Any]], qa_stats: QAStatistics) -> None:
        """
        Update Q&A statistics based on generated items.

        Args:
            qa_items: List of Q&A items
            qa_stats: Q&A statistics object to update
        """
        for item in qa_items:
            question_type = item['question_type']
            if question_type == 'factual':
                qa_stats.factual += 1
            elif question_type == 'multihop':
                qa_stats.multihop += 1
            elif question_type == 'trap':
                qa_stats.trap += 1

    @staticmethod
    def build_product_prompt(stats: Statistics, iteration: int) -> str:
        """
        Build user message for product generation based on current statistics.

        Args:
            stats: Current statistics
            iteration: Current iteration number

        Returns:
            User message string
        """
        if iteration == 0:
            return "This is the first run. Generate the first product data!"

        return f"""Generate the next diverse product!
        
Current dataset information:
- Total products: {stats.product_count}
- Domain distribution: {dict(stats.domains)}
- Existing product names: {stats.product_names}

Latest generated product:
{json.dumps(stats.latest_product, indent=2)}
"""

    @staticmethod
    def build_qa_prompt(product_json: str, qa_stats: QAStatistics) -> str:
        """
        Build user message for Q&A generation including statistics.

        Args:
            product_json: Product data as JSON string
            qa_stats: Current Q&A statistics

        Returns:
            User message string
        """
        distribution_info = f"""Current Q&A distribution:
- Factual: {qa_stats.factual}
- Multihop: {qa_stats.multihop}
- Trap: {qa_stats.trap}
- Total: {qa_stats.total}

"""
        return distribution_info + "Product data:\n" + product_json


class DataGenerator:
    """Main data generation orchestrator."""

    def __init__(self, config: GenerationConfig):
        """
        Initialize data generator.

        Args:
            config: Generation configuration
        """
        self.config = config
        self.llm_client = LLMClient(config)
        self.file_handler = FileHandler(config.output_dir)
        self.conversation_logger = ConversationLogger(config.log_dir)
        self.stats = Statistics()
        self.qa_stats = QAStatistics()

    def load_prompts(self) -> tuple[str, str]:
        """
        Load system prompts from files.

        Returns:
            Tuple of (products_prompt, questions_prompt)
        """
        products_prompt = self.file_handler.load_prompt(self.config.products_prompt_file)
        questions_prompt = self.file_handler.load_prompt(self.config.questions_prompt_file)
        return products_prompt, questions_prompt

    def generate_product(
        self,
        products_system_msg: str,
        iteration: int,
        pbar: Optional[tqdm] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a single product.

        Args:
            products_system_msg: System message for product generation
            iteration: Current iteration number
            pbar: Progress bar instance

        Returns:
            Generated product data or None if generation failed
        """
        # Build user message
        user_message = StatisticsTracker.build_product_prompt(self.stats, iteration)

        # Generate product
        if pbar:
            pbar.set_description(f"Generating product {iteration + 1}/{self.config.number_of_products}")

        try:
            seed = self.config.base_seed + iteration
            response, start_time, end_time, duration = self.llm_client.generate(
                products_system_msg, user_message, seed
            )
            content = response['message']['content']

            # Extract reasoning if available
            reasoning = response['message']['thinking']

            # Parse response
            data = json.loads(content)
            product = data.get('product', {})

            # Use iteration index as product_id
            product['product_id'] = str(iteration)

            # Log the conversation
            self.conversation_logger.log_conversation(
                conversation_type="product",
                product_id=product.get('product_id'),
                system_message=products_system_msg,
                user_message=user_message,
                response_content=content,
                reasoning=reasoning,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                seed=seed,
                temperature=self.config.temperature,
                model_name=self.config.model_name
            )

            # Save product file
            filename = f"{product['product_id']}-data.json"
            self.file_handler.save_text(filename, content)

            if pbar:
                pbar.write(f"✓ Saved product to {filename} (Generation time: {duration:.2f} ms)")

            # Update statistics
            self.stats.latest_product = product
            StatisticsTracker.update_product_stats(product, self.stats)

            return product

        except json.JSONDecodeError as e:
            if pbar:
                pbar.write(f"✗ Error: Could not parse product JSON response: {e}")
            return None
        except Exception as e:
            if pbar:
                pbar.write(f"✗ Error generating product: {e}")
            return None

    def generate_qa_data(
        self,
        product_id: str,
        product_json: str,
        questions_system_msg: str,
        iteration: int,
        pbar: Optional[tqdm] = None
    ) -> bool:
        """
        Generate Q&A data for a product.

        Args:
            product_id: Product identifier
            product_json: Product data as JSON string
            questions_system_msg: System message for Q&A generation
            iteration: Current iteration number
            pbar: Progress bar instance

        Returns:
            True if successful, False otherwise
        """
        # Build user message
        user_message = StatisticsTracker.build_qa_prompt(product_json, self.qa_stats)

        if pbar:
            pbar.set_description(f"Generating Q&A for {product_id}")

        try:
            seed = self.config.base_seed + iteration
            qa_response, start_time, end_time, duration = self.llm_client.generate(
                questions_system_msg, user_message, seed
            )
            qa_content = qa_response['message']['content']

            # Extract reasoning if available
            reasoning = qa_response['message']['thinking']

            # Log the conversation
            self.conversation_logger.log_conversation(
                conversation_type="qa",
                product_id=product_id,
                system_message=questions_system_msg,
                user_message=user_message,
                response_content=qa_content,
                reasoning=reasoning,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                seed=seed,
                temperature=self.config.temperature,
                model_name=self.config.model_name
            )

            # Parse Q&A response
            qa_items = json.loads(qa_content)
            questions_list = qa_items.get('questions', [])

            if not questions_list:
                if pbar:
                    pbar.write(f"✗ Warning: No questions generated for {product_id}")
                return False

            # Separate into questions and answers
            questions = []
            answers = []

            for item in questions_list:
                questions.append({
                    "id": item['id'],
                    "question": item['question'],
                    "question_type": item['question_type']
                })
                answers.append({
                    "id": item['id'],
                    "answer": item['answer']
                })

            # Save questions and answers files
            questions_filename = f"{product_id}-questions.json"
            answers_filename = f"{product_id}-answers.json"

            self.file_handler.save_json(questions_filename, questions)
            self.file_handler.save_json(answers_filename, answers)

            # Update Q&A statistics
            StatisticsTracker.update_qa_stats(questions_list, self.qa_stats)

            if pbar:
                pbar.write(f"✓ Saved {len(questions)} Q&A pairs for {product_id} (Generation time: {duration:.2f} ms)")

            return True

        except json.JSONDecodeError as e:
            if pbar:
                pbar.write(f"✗ Error: Could not parse Q&A JSON response: {e}")
            return False
        except Exception as e:
            if pbar:
                pbar.write(f"✗ Error generating Q&A: {e}")
            return False

    def run(self) -> int:
        """
        Run the data generation process.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            # Load prompts
            print("Loading system prompts...")
            products_system_msg, questions_system_msg = self.load_prompts()
            print("✓ Prompts loaded successfully")

            # Generate products with progress bar
            print(f"\nGenerating {self.config.number_of_products} products...")
            with tqdm(
                total=self.config.number_of_products,
                desc="Generating products",
                unit="product"
            ) as pbar:
                for i in range(self.config.number_of_products):
                    # Generate product
                    product = self.generate_product(products_system_msg, i, pbar)

                    if product:
                        # Generate Q&A data
                        product_json = json.dumps({"product": product}, indent=2)
                        self.generate_qa_data(
                            product['product_id'],
                            product_json,
                            questions_system_msg,
                            i,
                            pbar
                        )

                    pbar.update(1)

            # Print summary
            print("\n" + "=" * 60)
            print("Generation Complete!")
            print("=" * 60)

        except Exception as e:
            print(f"\n✗ Fatal error: {e}")
            return 1


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic product data and Q&A pairs using LLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss:20b",
        help="Ollama model name (default: gpt-oss:20b)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of products to generate (default: 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for generated files (default: current directory)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for conversation logs (default: logs)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--think-mode",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="LLM thinking mode (default: medium)"
    )
    parser.add_argument(
        "--product-data-prompt",
        type=Path,
        help="The system prompt for product data creation"
    )
    parser.add_argument(
        "--questions-prompt",
        type=Path,
        help="The system prompt for question creation"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the script.

    Returns:
        Exit code
    """
    # Parse arguments
    args = parse_arguments()

    # Create configuration
    config = GenerationConfig(
        model_name=args.model,
        number_of_products=args.count,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        temperature=args.temperature,
        base_seed=args.seed,
        think_mode=args.think_mode,
        products_prompt_file=args.product_data_prompt,
        questions_prompt_file=args.questions_prompt
    )

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    # Run generator
    generator = DataGenerator(config)
    return generator.run()


if __name__ == "__main__":
    sys.exit(main())