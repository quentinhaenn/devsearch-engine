"""
entry point for the application
"""

import click
from src.search.engine import SearchEngine

@click.group()
def cli():
    """Command line interface for the search application."""
    pass

@cli.command()
@click.argument('query', type=str)
@click.option('--limit', default=10, help='Number of results to return')
@click.option('--type', default="simple", type=click.Choice({'simple', 'advanced', 'semantic'}),
              help='Type of search to perform')
def search(query, limit, type):
    """Perform a search with the given query."""
    engine = SearchEngine()
    results = engine.search(query, top_k=limit, search_type=type).get("results", [])
    if not results:
        click.echo("No results found.")
        return
    for idx, result in enumerate(results, start=1):
        click.echo(f"{idx}.\n Title: {result['title']}\n Content: {result['content']}\nScore: {result['score']}\n")

if __name__ == "__main__":
    cli()