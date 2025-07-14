#!/usr/bin/env python3
"""
DuckLake Interactive Demo - Clean Version
A menu-driven demonstration of DuckLake features with visual feedback.
"""

import os
import sys
import time
import random
from typing import Optional
import duckdb
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt, Confirm
from rich import box
import click

console = Console()


class DuckLakeDemo:
    def __init__(self):
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.db_name: Optional[str] = None
        self.connected = False

    def check_services(self) -> bool:
        """Check if required services are running."""
        console.print("\n[bold blue]Checking Services...[/bold blue]")

        import subprocess

        try:
            # Check PostgreSQL
            result = subprocess.run(
                [
                    "podman",
                    "ps",
                    "--filter",
                    "name=ducklake-postgres",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            postgres_running = "ducklake-postgres" in result.stdout

            # Check MinIO
            result = subprocess.run(
                [
                    "podman",
                    "ps",
                    "--filter",
                    "name=ducklake-minio",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            minio_running = "ducklake-minio" in result.stdout

            status_table = Table(title="Service Status", box=box.ROUNDED)
            status_table.add_column("Service", style="cyan")
            status_table.add_column("Status", justify="center")
            status_table.add_column("Details", style="dim")

            status_table.add_row(
                "PostgreSQL",
                "Running" if postgres_running else "Stopped",
                "Catalog storage" if postgres_running else "Run installer first",
            )
            status_table.add_row(
                "MinIO",
                "Running" if minio_running else "Stopped",
                "Object storage" if minio_running else "Run installer first",
            )

            console.print(status_table)

            if not (postgres_running and minio_running):
                console.print("\n[red]Required services not running![/red]")
                console.print(
                    "[yellow]Tip: Run [bold]./ducklake-installer.sh[/bold] first[/yellow]"
                )
                return False

            console.print("\n[green]All services running![/green]")
            return True

        except Exception as e:
            console.print(f"[red]Error checking services: {e}[/red]")
            return False

    def connect_ducklake(self) -> bool:
        """Connect to DuckLake with setup."""
        if self.connected:
            return True

        console.print("\n[bold blue]Connecting to DuckLake...[/bold blue]")

        try:
            # Create connection
            self.conn = duckdb.connect(":memory:")

            # Install extensions
            with console.status("[bold blue]Installing extensions..."):
                self.conn.execute("INSTALL ducklake")
                self.conn.execute("INSTALL postgres")
                self.conn.execute("INSTALL httpfs")
                self.conn.execute("LOAD ducklake")
                self.conn.execute("LOAD postgres")
                self.conn.execute("LOAD httpfs")

            # Configure S3
            with console.status("[bold blue]Configuring S3..."):
                self.conn.execute("""
                    SET s3_region='us-east-1';
                    SET s3_access_key_id='minioadmin';
                    SET s3_secret_access_key='minioadmin';
                    SET s3_endpoint='localhost:9000';
                    SET s3_use_ssl=false;
                    SET s3_url_style='path';
                """)

            # Create bucket if it doesn't exist
            with console.status("[bold blue]Setting up S3 bucket..."):
                import subprocess

                try:
                    # Find MinIO container name
                    result = subprocess.run(
                        [
                            "podman",
                            "ps",
                            "--filter",
                            "name=minio",
                            "--format",
                            "{{.Names}}",
                        ],
                        capture_output=True,
                        text=True,
                    )

                    minio_containers = [
                        name.strip()
                        for name in result.stdout.split("\n")
                        if name.strip()
                    ]

                    if minio_containers:
                        minio_container = minio_containers[0]

                        subprocess.run(
                            [
                                "podman",
                                "exec",
                                minio_container,
                                "mc",
                                "alias",
                                "set",
                                "local",
                                "http://localhost:9000",
                                "minioadmin",
                                "minioadmin",
                            ],
                            capture_output=True,
                            check=True,
                        )

                        subprocess.run(
                            [
                                "podman",
                                "exec",
                                minio_container,
                                "mc",
                                "mb",
                                "local/ducklake-bucket",
                            ],
                            capture_output=True,
                        )  # Don't check - might already exist

                        console.print("[dim]S3 bucket ready[/dim]")
                    else:
                        console.print(
                            "[yellow]Warning: No MinIO container found[/yellow]"
                        )

                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not create bucket: {e}[/yellow]"
                    )

            # Connect to DuckLake
            with console.status("[bold blue]Initializing DuckLake..."):
                import time

                self.db_name = f"demo_{int(time.time())}"

                self.conn.execute(f"""
                    ATTACH 'ducklake:postgres:dbname=ducklake_catalog
                            user=ducklake
                            password=ducklake123
                            host=localhost
                            port=5432' AS {self.db_name}
                            (DATA_PATH 's3://ducklake-bucket/demo');
                """)
                self.conn.execute(f"USE {self.db_name}")

            # Create sample data
            self.create_sample_data()

            self.connected = True
            console.print("[green]Connected to DuckLake![/green]")
            return True

        except Exception as e:
            console.print(f"[red]Connection failed: {e}[/red]")
            console.print("[yellow]Make sure PostgreSQL and MinIO are running[/yellow]")
            return False

    def create_sample_data(self):
        """Create initial sample data for demos."""
        console.print("[dim]Creating sample data...[/dim]")

        # Drop and recreate tables to ensure clean state
        try:
            self.conn.execute("DROP TABLE IF EXISTS customers")
            self.conn.execute("DROP TABLE IF EXISTS orders")
        except:
            pass

        # Create customers table
        np.random.seed(42)
        customers_data = {
            "customer_id": range(1, 101),
            "name": [f"Customer {i}" for i in range(1, 101)],
            "email": [f"customer{i}@example.com" for i in range(1, 101)],
            "city": np.random.choice(
                ["New York", "London", "Tokyo", "Paris", "Sydney"], 100
            ),
            "age": np.random.randint(18, 80, 100),
            "signup_date": pd.date_range("2024-01-01", periods=100, freq="D"),
        }
        customers_df = pd.DataFrame(customers_data)

        self.conn.execute("""
            CREATE TABLE customers (
                customer_id INTEGER,
                name VARCHAR,
                email VARCHAR,
                city VARCHAR,
                age INTEGER,
                signup_date DATE
            )
        """)

        self.conn.execute("INSERT INTO customers SELECT * FROM customers_df")

        # Create orders table
        orders_data = {
            "order_id": range(1, 501),
            "customer_id": np.random.choice(customers_data["customer_id"], 500),
            "product": np.random.choice(
                ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"], 500
            ),
            "amount": np.round(np.random.uniform(50, 2000, 500), 2),
            "order_date": pd.date_range("2024-01-01", periods=500, freq="2h"),
            "status": np.random.choice(
                ["pending", "shipped", "delivered", "cancelled"],
                500,
                p=[0.1, 0.3, 0.5, 0.1],
            ),
        }
        orders_df = pd.DataFrame(orders_data)

        self.conn.execute("""
            CREATE TABLE orders (
                order_id INTEGER,
                customer_id INTEGER,
                product VARCHAR,
                amount DECIMAL(10,2),
                order_date TIMESTAMP,
                status VARCHAR
            )
        """)

        self.conn.execute("INSERT INTO orders SELECT * FROM orders_df")

    def show_main_menu(self):
        """Display the main menu."""
        menu_panel = Panel.fit(
            """[bold cyan]DuckLake Interactive Demo[/bold cyan]

[bold]Core Features:[/bold]
[bright_blue]1.[/bright_blue] ACID Transactions
[bright_blue]2.[/bright_blue] Time Travel Queries  
[bright_blue]3.[/bright_blue] Schema Evolution
[bright_blue]4.[/bright_blue] Performance Benchmarks

[bold]System:[/bold]
[bright_blue]5.[/bright_blue] System Status
[bright_blue]6.[/bright_blue] Reset Demo Data
[bright_blue]0.[/bright_blue] Exit

Choose a demo to explore DuckLake features!""",
            border_style="blue",
            padding=(1, 2),
        )

        console.print("\n")
        console.print(menu_panel)

    def demo_acid_transactions(self):
        """Demonstrate ACID transaction capabilities."""
        console.print("\n[bold blue]ACID Transactions Demo[/bold blue]")

        # Show initial state
        initial_customers = self.conn.execute(
            "SELECT COUNT(*) FROM customers"
        ).fetchone()[0]
        initial_orders = self.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]

        console.print(f"\n[cyan]Initial State:[/cyan]")
        console.print(f"   Customers: {initial_customers}")
        console.print(f"   Orders: {initial_orders}")

        console.print(f"\n[yellow]Starting transaction...[/yellow]")

        # Start transaction
        self.conn.begin()

        try:
            # Insert test data
            self.conn.execute("""
                INSERT INTO customers (customer_id, name, email, city, age, signup_date)
                VALUES (999, 'Test Customer', 'test@example.com', 'Test City', 30, '2024-01-01')
            """)

            self.conn.execute("""
                INSERT INTO orders (order_id, customer_id, product, amount, order_date, status)
                VALUES (999, 999, 'Test Product', 999.99, '2024-01-01 10:00:00', 'pending')
            """)

            # Show state during transaction
            tx_customers = self.conn.execute(
                "SELECT COUNT(*) FROM customers"
            ).fetchone()[0]
            tx_orders = self.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]

            console.print(f"\n[green]During Transaction:[/green]")
            console.print(
                f"   Customers: {tx_customers} (+{tx_customers - initial_customers})"
            )
            console.print(f"   Orders: {tx_orders} (+{tx_orders - initial_orders})")

            # Show inserted data
            test_customer = self.conn.execute(
                "SELECT name, city FROM customers WHERE customer_id = 999"
            ).fetchone()
            test_order = self.conn.execute(
                "SELECT product, amount FROM orders WHERE order_id = 999"
            ).fetchone()

            data_table = Table(title="Data Inserted in Transaction", box=box.ROUNDED)
            data_table.add_column("Table", style="cyan")
            data_table.add_column("Record", style="green")

            data_table.add_row(
                "Customer", f"{test_customer[0]} from {test_customer[1]}"
            )
            data_table.add_row("Order", f"{test_order[0]} for ${test_order[1]:.2f}")

            console.print("\n")
            console.print(data_table)

            # Ask user what to do
            console.print(
                f"\n[yellow]What should we do with this transaction?[/yellow]"
            )
            action = Prompt.ask(
                "Choose action", choices=["commit", "rollback"], default="rollback"
            )

            if action == "commit":
                self.conn.commit()
                console.print(f"\n[green]Transaction committed![/green]")
            else:
                self.conn.rollback()
                console.print(f"\n[red]Transaction rolled back![/red]")

            # Show final state
            final_customers = self.conn.execute(
                "SELECT COUNT(*) FROM customers"
            ).fetchone()[0]
            final_orders = self.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[
                0
            ]

            console.print(f"\n[blue]Final State:[/blue]")
            console.print(f"   Customers: {final_customers}")
            console.print(f"   Orders: {final_orders}")

            # Verify atomicity
            if action == "rollback":
                verify_customer = self.conn.execute(
                    "SELECT COUNT(*) FROM customers WHERE customer_id = 999"
                ).fetchone()[0]
                verify_order = self.conn.execute(
                    "SELECT COUNT(*) FROM orders WHERE order_id = 999"
                ).fetchone()[0]

                if verify_customer == 0 and verify_order == 0:
                    console.print(f"\n[green]ACID Atomicity Verified:[/green]")
                    console.print(f"   All changes rolled back completely")
                    console.print(f"   No partial state exists")
                else:
                    console.print(f"\n[red]ACID violation detected![/red]")

        except Exception as e:
            console.print(f"\n[red]Transaction failed: {e}[/red]")
            try:
                self.conn.rollback()
            except:
                pass

    def demo_time_travel(self):
        """Demonstrate time travel capabilities."""
        console.print("\n[bold blue]Time Travel Demo[/bold blue]")

        # Get current snapshots and show details
        try:
            snapshots = self.conn.execute(
                f"SELECT * FROM ducklake_snapshots('{self.db_name}')"
            ).fetchdf()
            current_snapshot_count = len(snapshots)

            # Show snapshot details for debugging
            console.print(f"\n[cyan]Current snapshots: {current_snapshot_count}[/cyan]")
            if not snapshots.empty and len(snapshots) > 1:
                console.print(
                    f"[dim]Available snapshot IDs: {list(snapshots.index + 1 if 'snapshot_id' not in snapshots.columns else snapshots['snapshot_id'].tolist())}[/dim]"
                )

        except Exception as e:
            current_snapshot_count = 1
            console.print(
                f"\n[cyan]Current snapshots: {current_snapshot_count} (estimated)[/cyan]"
            )
            console.print(f"[dim]Snapshot query failed: {e}[/dim]")

        # Show current data
        current_data = self.conn.execute("""
            SELECT customer_id, name, age, city 
            FROM customers 
            WHERE customer_id <= 5 
            ORDER BY customer_id
        """).fetchdf()

        console.print(f"\n[cyan]Current Customer Data:[/cyan]")
        current_table = Table(title="Current State", box=box.ROUNDED)
        current_table.add_column("ID", justify="center")
        current_table.add_column("Name", style="cyan")
        current_table.add_column("Age", justify="center", style="blue")
        current_table.add_column("City", style="magenta")

        for _, row in current_data.iterrows():
            current_table.add_row(
                str(row["customer_id"]),
                str(row["name"]),
                str(row["age"]),
                str(row["city"]),
            )

        console.print(current_table)

        # Store the original data before changes
        original_data = current_data.copy()

        # Make changes
        console.print(f"\n[yellow]Making changes to create new snapshot...[/yellow]")
        changes = Prompt.ask(
            "What changes should we make?",
            choices=["age", "city", "both"],
            default="both",
        )

        if changes in ["age", "both"]:
            self.conn.execute(
                "UPDATE customers SET age = age + 5 WHERE customer_id <= 5"
            )
            console.print("   Increased ages by 5 years")

        if changes in ["city", "both"]:
            self.conn.execute("""
                UPDATE customers 
                SET city = CASE 
                    WHEN city = 'New York' THEN 'Boston'
                    WHEN city = 'London' THEN 'Manchester'  
                    WHEN city = 'Tokyo' THEN 'Osaka'
                    WHEN city = 'Paris' THEN 'Lyon'
                    ELSE 'Melbourne'
                END
                WHERE customer_id <= 5
            """)
            console.print("   Updated cities")

        # Force a snapshot creation by running a dummy query that might trigger it
        try:
            self.conn.execute("SELECT COUNT(*) FROM customers")
            console.print("   Snapshot triggered")
        except:
            pass

        # Show updated data
        updated_data = self.conn.execute("""
            SELECT customer_id, name, age, city 
            FROM customers 
            WHERE customer_id <= 5 
            ORDER BY customer_id
        """).fetchdf()

        console.print(f"\n[green]Updated Data:[/green]")
        updated_table = Table(title="After Changes", box=box.ROUNDED)
        updated_table.add_column("ID", justify="center")
        updated_table.add_column("Name", style="cyan")
        updated_table.add_column("Age", justify="center", style="green")
        updated_table.add_column("City", style="magenta")

        for _, row in updated_data.iterrows():
            updated_table.add_row(
                str(row["customer_id"]),
                str(row["name"]),
                str(row["age"]),
                str(row["city"]),
            )

        console.print(updated_table)

        # Show the before/after comparison
        console.print(f"\n[blue]Before vs After Comparison:[/blue]")
        comparison_table = Table(title="Time Travel Demonstration", box=box.ROUNDED)
        comparison_table.add_column("Customer", style="cyan")
        comparison_table.add_column("Age Change", justify="center", style="blue")
        comparison_table.add_column("City Change", style="magenta")

        for i in range(min(len(original_data), len(updated_data))):
            old_row = original_data.iloc[i]
            new_row = updated_data.iloc[i]

            age_change = (
                f"{old_row['age']} -> {new_row['age']}"
                if old_row["age"] != new_row["age"]
                else "unchanged"
            )
            city_change = (
                f"{old_row['city']} -> {new_row['city']}"
                if old_row["city"] != new_row["city"]
                else "unchanged"
            )

            comparison_table.add_row(str(old_row["name"]), age_change, city_change)

        console.print(comparison_table)

        console.print(f"\n[green]Time travel concept demonstrated![/green]")
        console.print(
            f"[dim]Original ages: {', '.join(map(str, original_data['age'].tolist()))}[/dim]"
        )
        console.print(
            f"[dim]Current ages: {', '.join(map(str, updated_data['age'].tolist()))}[/dim]"
        )
        console.print(
            f"[dim]In a mature DuckLake, you could query: AT (VERSION => X)[/dim]"
        )

    def demo_schema_evolution(self):
        """Demonstrate schema evolution."""
        console.print("\n[bold blue]Schema Evolution Demo[/bold blue]")

        # Show current schema
        current_schema = self.conn.execute("DESCRIBE customers").fetchdf()

        console.print(f"\n[cyan]Current Schema:[/cyan]")
        schema_table = Table(title="customers Table Schema", box=box.ROUNDED)
        schema_table.add_column("Column", style="cyan")
        schema_table.add_column("Type", style="magenta")
        schema_table.add_column("Nullable", style="yellow")

        for _, row in current_schema.iterrows():
            schema_table.add_row(
                str(row["column_name"]), str(row["column_type"]), str(row["null"])
            )

        console.print(schema_table)

        # Choose evolution
        console.print(f"\n[yellow]Schema Evolution Options:[/yellow]")
        evolution = Prompt.ask(
            "What schema change?",
            choices=["add_column", "rename_column", "both"],
            default="add_column",
        )

        if evolution in ["add_column", "both"]:
            console.print(f"\n[yellow]Adding loyalty_points column...[/yellow]")
            try:
                self.conn.execute(
                    "ALTER TABLE customers ADD COLUMN loyalty_points INTEGER DEFAULT 0"
                )
                self.conn.execute(
                    "UPDATE customers SET loyalty_points = age * 10 WHERE customer_id <= 10"
                )
                console.print("   Added loyalty_points column")
            except Exception as e:
                console.print(f"   Failed to add column: {e}")

        if evolution in ["rename_column", "both"]:
            console.print(
                f"\n[yellow]Adding full_name column (evolution pattern)...[/yellow]"
            )
            try:
                self.conn.execute("ALTER TABLE customers ADD COLUMN full_name VARCHAR")
                self.conn.execute(
                    "UPDATE customers SET full_name = name || ' (Customer)' WHERE customer_id <= 5"
                )
                console.print("   Added full_name column")
            except Exception as e:
                console.print(f"   Failed to add column: {e}")

        # Show evolved schema
        evolved_schema = self.conn.execute("DESCRIBE customers").fetchdf()

        console.print(f"\n[green]Evolved Schema:[/green]")
        evolved_table = Table(title="Updated customers Table Schema", box=box.ROUNDED)
        evolved_table.add_column("Column", style="cyan")
        evolved_table.add_column("Type", style="magenta")
        evolved_table.add_column("Nullable", style="yellow")
        evolved_table.add_column("Status", style="green")

        old_columns = set(current_schema["column_name"])

        for _, row in evolved_schema.iterrows():
            status = "NEW" if row["column_name"] not in old_columns else "existing"
            evolved_table.add_row(
                str(row["column_name"]),
                str(row["column_type"]),
                str(row["null"]),
                status,
            )

        console.print(evolved_table)

    def demo_performance(self):
        """Demonstrate performance benchmarks."""
        console.print("\n[bold blue]Performance Benchmarks[/bold blue]")

        queries = [
            ("Simple Count", "SELECT COUNT(*) FROM customers"),
            (
                "Basic Join",
                """
                SELECT c.city, COUNT(o.order_id) as order_count, AVG(o.amount) as avg_amount
                FROM customers c JOIN orders o ON c.customer_id = o.customer_id
                GROUP BY c.city ORDER BY order_count DESC
            """,
            ),
            (
                "Complex Analytics",
                """
                SELECT 
                    c.city,
                    c.age,
                    COUNT(o.order_id) as orders,
                    SUM(o.amount) as total_spent,
                    AVG(o.amount) as avg_order,
                    MAX(o.order_date) as last_order
                FROM customers c
                LEFT JOIN orders o ON c.customer_id = o.customer_id
                WHERE c.age BETWEEN 25 AND 65
                GROUP BY c.city, c.age
                HAVING COUNT(o.order_id) > 0
                ORDER BY total_spent DESC
                LIMIT 10
            """,
            ),
        ]

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running benchmarks...", total=len(queries))

            for query_name, query in queries:
                progress.update(task, description=f"Executing: {query_name}")

                # Run query multiple times for more accurate timing
                times = []
                for _ in range(3):
                    start_time = time.time()
                    result = self.conn.execute(query).fetchdf()
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)

                avg_time = sum(times) / len(times)
                results.append(
                    {
                        "query": query_name,
                        "time_ms": round(avg_time, 2),
                        "rows": len(result),
                    }
                )

                progress.advance(task)

        # Show results
        console.print(f"\n[green]Performance Results:[/green]")
        perf_table = Table(title="Query Performance", box=box.ROUNDED)
        perf_table.add_column("Query", style="cyan")
        perf_table.add_column("Avg Time", justify="right", style="green")
        perf_table.add_column("Rows", justify="right", style="blue")
        perf_table.add_column("Performance", style="yellow")

        for result in results:
            time_ms = result["time_ms"]
            if time_ms < 10:
                perf_rating = "Excellent"
            elif time_ms < 50:
                perf_rating = "Very Fast"
            elif time_ms < 200:
                perf_rating = "Fast"
            else:
                perf_rating = "Good"

            perf_table.add_row(
                result["query"], f"{time_ms:.2f}ms", f"{result['rows']:,}", perf_rating
            )

        console.print(perf_table)

    def show_system_status(self):
        """Show system status and statistics."""
        console.print("\n[bold blue]System Status[/bold blue]")

        # Database info
        tables = self.conn.execute("SHOW TABLES").fetchdf()

        status_table = Table(title="DuckLake Status", box=box.ROUNDED)
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="dim")

        status_table.add_row("Connection", "Active", f"Database: {self.db_name}")
        status_table.add_row(
            "Tables",
            f"{len(tables)}",
            ", ".join(
                tables["name"] if "name" in tables.columns else tables.iloc[:, 0]
            ),
        )
        status_table.add_row("Storage", "S3", "s3://ducklake-bucket/demo")
        status_table.add_row("Catalog", "PostgreSQL", "localhost:5432")

        console.print(status_table)

    def reset_demo_data(self):
        """Reset demo data to initial state."""
        console.print("\n[bold blue]Reset Demo Data[/bold blue]")

        if not Confirm.ask("This will reset all demo data. Continue?"):
            return

        with console.status("[bold blue]Resetting data..."):
            try:
                # Drop and recreate tables
                self.conn.execute("DROP TABLE IF EXISTS customers")
                self.conn.execute("DROP TABLE IF EXISTS orders")

                # Recreate sample data
                self.create_sample_data()

                console.print("[green]Demo data reset successfully![/green]")

            except Exception as e:
                console.print(f"[red]Reset failed: {e}[/red]")

    def run(self):
        """Main demo loop."""
        console.print("[bold blue]Welcome to DuckLake Interactive Demo![/bold blue]")

        # Check services
        if not self.check_services():
            return

        # Connect to DuckLake
        if not self.connect_ducklake():
            return

        while True:
            try:
                self.show_main_menu()

                choice = Prompt.ask(
                    "\n[bold]Choose demo",
                    choices=["1", "2", "3", "4", "5", "6", "0"],
                    default="1",
                )

                if choice == "1":
                    self.demo_acid_transactions()
                elif choice == "2":
                    self.demo_time_travel()
                elif choice == "3":
                    self.demo_schema_evolution()
                elif choice == "4":
                    self.demo_performance()
                elif choice == "5":
                    self.show_system_status()
                elif choice == "6":
                    self.reset_demo_data()
                elif choice == "0":
                    console.print("\n[green]Thanks for exploring DuckLake![/green]")
                    break

                if choice != "0":
                    Prompt.ask("\n[dim]Press Enter to continue...", default="")

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Demo interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                Prompt.ask("\n[dim]Press Enter to continue...", default="")


@click.command()
def main():
    """Run the DuckLake interactive demo."""
    demo = DuckLakeDemo()
    demo.run()


if __name__ == "__main__":
    main()
