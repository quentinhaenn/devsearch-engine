"""
entry point for the application
"""
import logging
import click
from src.search.engine import SearchEngine
from src.data.loader import DataLoader

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Command line interface for the search application."""
    pass

@cli.command()
@click.argument('query', type=str)
@click.option('-k',"--top-k", default=10, help='Number of results to return')
@click.option("--no-rerank", is_flag=True,default=False, help='Disable reranking of results')
@click.option("--filter-source", multiple=True, help='Filter results by source type (e.g., official, documentation, blog)')
@click.option("--filter-language", multiple=True, help='Filter results by language (e.g., en, fr, de)')
@click.option("--date-range", nargs=2, type=str, help='Filter results by date range (start_date end_date). If one date is provided, it will be used as the start date and the end date will be today.')
def search(query: str,
        top_k: int,
        no_rerank: bool,
        filter_source: list[str],
        filter_language: list[str],
        date_range: list[str]) -> None:
    """Perform a search with the given query."""
    engine = SearchEngine()
    
    filter_source = list(filter_source) if filter_source else None
    filter_language = list(filter_language) if filter_language else None

    try:
        results = engine.search(query, top_k=top_k, no_rerank=no_rerank, filter_source=filter_source, filter_language=filter_language, date_range=date_range)

        if not results:
            click.echo("No results found.")
            return
        
    
    except Exception as e:
        logger.error(f"An error occurred during search: {e}")
        click.echo("An error occurred during search. Please check the logs for more details.")
        return

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Found {len(results.original_results)} results for query: {query}")
    click.echo(f"Search time taken: {results.search_time} seconds")

    if not no_rerank:
        click.echo(f"Reranking time taken: {results.reranking_time} seconds")
        click.echo(f"Displaying top {top_k} results:")
        engine.explain_results(results, query)
    
    else:
        click.echo("Reranking is disabled. Displaying raw results:")
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            table = Table(title="Search Results", show_header=True, header_style="bold magenta")
            table.add_column("Rank", style="dim", width=6)
            table.add_column("Score", style="dim", width=10)
            table.add_column("Title", style="bold")
            table.add_column("Source", style="dim")
            table.add_column("Language", style="dim")
            table.add_column("Snippet", style="dim")
            table.add_column("URL/Path", style="dim")

            for i, result in enumerate(results.original_results[:top_k]):
                table.add_row(
                    str(i + 1),
                    f"{result.get('score', 0.0):.2f}",
                    result.get("title", ""),
                    result.get("source", ""),
                    result.get("language", ""),
                    result.get("snippet", ""),
                    result.get("url") or result.get("path", "")
                )
            console.print(table)
        except ImportError:
            click.echo("Rich library is not installed. Please install it to view the results.")
            return

        except Exception as e:
            logger.error(f"Error occurred while displaying results: {e}")
            click.echo("An error occurred while displaying results. Please check the logs for more details.")

if __name__ == "__main__":
    cli()